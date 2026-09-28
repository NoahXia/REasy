import unittest
from types import SimpleNamespace

from file_handlers.motion.preview.event_defaults import missing_event_fields


class Registry:
    def find_type_by_name(self, name):
        return ({
            "app.motion_track.PlayerTrackingTarget": {"fields": [
                {"name": "_MotionTransRateXZ", "original_type": "System.Single"},
                {"name": "MotionTransRate", "original_type": "Rates"},
                {"name": "RotLimit_Deg", "original_type": "System.Single"},
                {"name": "UnknownField", "original_type": "System.Single"},
            ]},
            "Rates": {"fields": [{"name": "x"}, {"name": "z"}]},
        }.get(name), None)


class EventDefaultTests(unittest.TestCase):
    def rows(self, properties=(), game="wots"):
        return dict(missing_event_fields("PlayerTrackingTarget", properties, Registry(), game=game))

    def test_missing_and_unknown_values(self):
        rows = self.rows()
        self.assertTrue(rows["MotionTransRateXZ"].startswith("1.0"))
        self.assertTrue(rows["MotionTransRate.x"].startswith("1.0"))
        self.assertTrue(rows["RotLimit_Deg"].startswith("0.0"))
        self.assertIn("observed constructor", rows["RotLimit_Deg"])
        self.assertIn("unresolved", rows["UnknownField"])

    def test_explicit_nested_and_backing_fields_take_precedence(self):
        props = (SimpleNamespace(name="<MotionTransRateXZ>k__BackingField", children=[]),
                 SimpleNamespace(name="MotionTransRate", children=[SimpleNamespace(name="x", children=[])]))
        rows = self.rows(props)
        self.assertNotIn("MotionTransRateXZ", rows)
        self.assertNotIn("MotionTransRate.x", rows)
        self.assertIn("MotionTransRate.z", rows)
        self.assertEqual(len(props[1].children), 1)

    def test_game_isolation_and_missing_metadata(self):
        self.assertIn("unresolved", self.rows(game="dmc5")["MotionTransRateXZ"])
        self.assertEqual(missing_event_fields("Unknown", (), Registry()), ())
        self.assertEqual(missing_event_fields("PlayerTrackingTarget", (), None), ())


if __name__ == "__main__":
    unittest.main()
