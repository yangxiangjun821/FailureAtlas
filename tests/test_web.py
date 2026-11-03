from failure_atlas.web import build_dashboard_payload, dashboard, index, projects


def test_dashboard_payload_contains_comparison() -> None:
    payload = build_dashboard_payload()
    assert payload["clean_accuracy_after"] > payload["clean_accuracy_before"]
    assert len(payload["experiments"]) == 5
    assert len(payload["actions"]) == 3


def test_dashboard_entry_and_api_function() -> None:
    response = index()
    assert response.status_code == 200
    assert response.path.name == "index.html"
    assert dashboard()["dataset"] == "Fashion-MNIST"
    assert projects()["projects"][0]["status"] == "analyzed"
