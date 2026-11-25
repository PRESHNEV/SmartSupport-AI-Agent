def test_ticket_creation(tmp_path, monkeypatch):
    import app.agent as mod
    monkeypatch.setattr(mod, "DB", str(tmp_path / "test.db"))
    a = mod.Agent()
    result = a.run("Please create a support ticket for my purchase issue")
    assert result["intent"] == "create_ticket"
    assert result["tool_calls"][0]["ticket_id"] == 1
