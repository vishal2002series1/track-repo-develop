# test_episodic_memory.py
"""
Standalone smoke test for src/memory/episodic.py.

Run from the repo root:
    python test_episodic_memory.py

Should print ✅ lines and a final summary. Safe to re-run.
"""
import os
import sys

# Make repo-root imports work when run directly.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.memory.episodic import EpisodicMemory


def main():
    print("🔧 Initializing EpisodicMemory (this loads MiniLM, may take a few seconds)...")
    mem = EpisodicMemory()
    print(f"   Collection currently holds {mem.count()} episode(s).")

    # Start from a clean slate so the test is deterministic.
    mem.clear()
    assert mem.count() == 0
    print("🧹 Cleared collection.")

    # ---- Seed a mix of episodes across users / sessions ---------------------
    print("\n📝 Recording 5 episodes...")
    e1 = mem.record(
        session_id="s_alice_1", turn_idx=0,
        user_prompt="Show me clients with high concentration risk in their portfolio.",
        final_answer="Top 3 clients with overexposure: ...",
        plan={"mode": "single", "steps": [{"workflow_id": "WF_002"}]},
        confidence="high",
        tenant_id="ey-wealth", user_id="advisor_alice",
    )
    e2 = mem.record(
        session_id="s_alice_1", turn_idx=1,
        user_prompt="Now draft meeting agendas for those top 3 clients.",
        final_answer="Agenda for client A: ... Agenda for client B: ...",
        plan={"mode": "single", "steps": [{"workflow_id": "WF_003"}]},
        confidence="high",
        tenant_id="ey-wealth", user_id="advisor_alice",
    )
    e3 = mem.record(
        session_id="s_bob_1", turn_idx=0,
        user_prompt="Draft an agenda for the Johnson family review.",
        final_answer="Agenda: 1) Portfolio review ...",
        plan={"mode": "single", "steps": [{"workflow_id": "WF_003"}]},
        confidence="high",
        tenant_id="ey-wealth", user_id="advisor_bob",
    )
    e4 = mem.record(
        session_id="s_carol_1", turn_idx=0,
        user_prompt="What changed for Davidson household since last review?",
        final_answer="Since last review: ...",
        plan={"mode": "single", "steps": [{"workflow_id": "WF_003"}]},
        confidence="medium",
        tenant_id="ey-wealth", user_id="advisor_carol",
    )
    e5 = mem.record(
        session_id="s_dan_1", turn_idx=0,
        user_prompt="Show concentration risk in my client book.",
        final_answer="...",
        plan={"mode": "single", "steps": [{"workflow_id": "WF_002"}]},
        confidence="high",
        tenant_id="partner-firm-x", user_id="advisor_dan",
    )
    print(f"   Recorded ids: {[e1, e2, e3, e4, e5]}")
    print(f"   Total episodes now: {mem.count()}")
    assert mem.count() == 5

    # ---- Test 1: recent() returns session episodes newest-first -------------
    print("\n🧪 Test 1: recent(s_alice_1, k=3)")
    recent = mem.recent("s_alice_1", k=3)
    assert len(recent) == 2, f"expected 2, got {len(recent)}"
    assert recent[0]["metadata"]["turn_idx"] == 1, "newest turn should be first"
    assert recent[1]["metadata"]["turn_idx"] == 0
    print(f"   ✅ Got {len(recent)} episodes for s_alice_1, newest first.")

    # ---- Test 2: similar() scoped by user_id --------------------------------
    print("\n🧪 Test 2: similar('top concentration-risk clients then draft agendas',")
    print("                   user_id=advisor_alice)")
    sim = mem.similar(
        prompt="For my top concentration-risk clients, draft meeting agendas for each.",
        k=3, tenant_id="ey-wealth", user_id="advisor_alice",
    )
    sim_ids = [r["id"] for r in sim]
    print(f"   Returned ids: {sim_ids}")
    print(f"   Distances:    {[round(r['distance'], 4) for r in sim]}")
    # Must contain only Alice's episodes; must NOT contain Bob/Carol/Dan.
    for r in sim:
        assert r["metadata"]["user_id"] == "advisor_alice", \
            f"user_id leak: {r['metadata']}"
    print(f"   ✅ All {len(sim)} hits belong to advisor_alice.")

    # ---- Test 3: similar() scoped by tenant only ----------------------------
    print("\n🧪 Test 3: similar('draft meeting agenda', tenant_id=ey-wealth)")
    sim = mem.similar(
        prompt="Draft a meeting agenda for my next client review.",
        k=5, tenant_id="ey-wealth",
    )
    tenants = {r["metadata"]["tenant_id"] for r in sim}
    assert tenants == {"ey-wealth"}, f"tenant leak: {tenants}"
    print(f"   ✅ All {len(sim)} hits scoped to ey-wealth (no partner-firm-x leak).")

    # ---- Test 4: cross-tenant isolation -------------------------------------
    print("\n🧪 Test 4: similar('concentration risk', tenant_id=partner-firm-x)")
    sim = mem.similar(
        prompt="concentration risk in book of business",
        k=5, tenant_id="partner-firm-x",
    )
    assert all(r["metadata"]["tenant_id"] == "partner-firm-x" for r in sim)
    print(f"   ✅ Cross-tenant isolation holds ({len(sim)} hit(s), only partner-firm-x).")

    # ---- Test 5: exclude_session_id -----------------------------------------
    print("\n🧪 Test 5: similar(..., exclude_session_id=s_alice_1)")
    sim = mem.similar(
        prompt="concentration risk and agendas",
        k=5, tenant_id="ey-wealth", exclude_session_id="s_alice_1",
    )
    sessions = {r["metadata"]["session_id"] for r in sim}
    assert "s_alice_1" not in sessions, "current session leaked into similar()"
    print(f"   ✅ s_alice_1 excluded. Returned sessions: {sessions}")

    print("\n🎉 All episodic memory tests passed.")
    print(f"   Persisted at: {os.path.abspath('data_local/episodic_chroma')}")


if __name__ == "__main__":
    main()