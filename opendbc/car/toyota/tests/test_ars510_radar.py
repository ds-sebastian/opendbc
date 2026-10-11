import unittest
import zlib

from opendbc.car import structs
from opendbc.car.toyota.ars510_radar import IDLE_SLOT, RECORD_LEN, SLOT_LEN, SLOT_START, SLOTS, RadarInterface

EGO_SPEED = 18.0  # m/s
SPEED_FRAME = bytes(5) + round(EGO_SPEED * 3.6 / 0.01).to_bytes(2, "big") + bytes(1)  # 0xB4


def set_bits(slot: bytearray, start: int, length: int, value: int) -> None:
  word = int.from_bytes(slot, "little") & ~(((1 << length) - 1) << start)
  slot[:] = (word | value << start).to_bytes(len(slot), "little")


def record(age: int, d_rel: float = 40.0, y_rel: float = 1.5, v_ground: float = 20.0, unc: int = 5) -> bytes:
  slot = bytearray(IDLE_SLOT)
  for start, length, value in ((24, 7, age), (32, 12, round(d_rel * 16) + 160), (44, 12, round(y_rel / 0.015) + 2048),
                               (64, 10, round(v_ground / 0.149 + 510.5)), (240, 7, unc), (56, 7, 1)):
    set_bits(slot, start, length, value)
  rec = bytearray(RECORD_LEN)
  rec[0] = 0xE4
  for s in range(SLOTS):
    rec[SLOT_START + SLOT_LEN * s:SLOT_START + SLOT_LEN * (s + 1)] = slot if s == 3 else IDLE_SLOT
  rec[737:741] = zlib.crc32(bytes(rec[1:737])).to_bytes(4, "little")
  return bytes(rec)


def frames(rec: bytes) -> list[bytes]:
  """One 742-byte record as 106 consecutive frames on 0x80 (first frame 0x12, then 0x2N with N wrapping every 16)."""
  return [bytes([0x12]) + rec[0:7]] + [bytes([0x20 | (k & 0xF)]) + rec[7 * k:7 * k + 7] for k in range(1, len(rec) // 7)]


class TestArs510Radar(unittest.TestCase):
  def setUp(self):
    self.ri = RadarInterface(structs.CarParams.new_message())
    self.t = 0

  def step(self, radar_frames):
    """One card.py update (100 Hz): ego speed on bus 0 and part of a record on bus 1."""
    self.t += 10_000_000
    return self.ri.update([(self.t, [(0xB4, SPEED_FRAME, 0)] + [(0x80, f, 1) for f in radar_frames])])

  def send(self, rec):
    """A record over 6 updates, as at the radar's 60 ms cycle; returns the report after the last frame."""
    fr = frames(rec)
    outs = [self.step(fr[j * 18:(j + 1) * 18]) for j in range(6)]
    assert all(o is None or (len(o.points) == 0 and not o.errors.radarUnavailableTemporary) for o in outs[:5])  # radarless at 20 Hz before the first record
    return outs[5]

  def test_young_tracks_held_back_then_published(self):
    reports = [self.send(record(age)) for age in range(55, 75)]
    assert all(len(r.points) == 0 for r in reports[:5])  # age < 60
    pts = [r.points[0] for r in reports[5:]]
    assert len({p.trackId for p in pts}) == 1
    p = pts[-1]
    assert abs(p.yRel - 1.5) < 1e-6
    assert abs(p.dRel - 40.0) < 2.5  # velocity-aided range settles a little away from a range held fixed while vRel != 0
    assert abs(p.vRel - (20.0 - EGO_SPEED)) < 0.1

  def test_crc_failure_dropped(self):
    rec = bytearray(record(80))
    rec[100] ^= 0xFF
    assert self.send(bytes(rec)) is None

  def test_silent_radar(self):
    for age in range(60, 70):
      self.send(record(age))
    outs = [self.step([]) for _ in range(100)]
    errs = [o for o in outs if o is not None]
    assert errs and all(o.errors.radarUnavailableTemporary and len(o.points) == 0 for o in errs)
    first = next(i for i, o in enumerate(outs) if o is not None)
    assert 40 <= first <= 60  # ~0.5 s after the last record, then 20 Hz


if __name__ == "__main__":
  unittest.main()
