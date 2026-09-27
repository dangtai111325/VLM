from rt_lcod.inference.temporal import SimpleTargetTracker


def test_track_id_stays_on_overlap():
    tracker = SimpleTargetTracker(iou_gate=0.2, max_missed=2)
    first = tracker.update([((0, 0, 20, 20), 0.9)])
    assert first is not None
    track_id = first.track_id
    second = tracker.update([((2, 1, 22, 21), 0.8)])
    assert second is not None
    assert second.track_id == track_id


def test_track_drops_after_misses():
    tracker = SimpleTargetTracker(iou_gate=0.2, max_missed=1)
    tracker.update([((0, 0, 20, 20), 0.9)])
    tracker.update([])
    tracker.update([])
    assert tracker.state is None
