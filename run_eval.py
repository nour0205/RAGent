import json
import requests

BASE_URL = "http://127.0.0.1:8000"


def normalize_answer(text: str) -> str:
    return text.strip().lower()


def is_refusal(text: str) -> bool:
    return normalize_answer(text) == "i don't know."


with open("eval_cases.json", "r", encoding="utf-8") as f:
    cases = json.load(f)

plan_pass = 0
api_route_pass = 0
answer_pass = 0
sources_presence_pass = 0
source_docs_pass = 0

for i, case in enumerate(cases, 1):
    question = case["question"]

    # -------------------------
    # Planner check
    # -------------------------
    plan_resp = requests.post(
        f"{BASE_URL}/debug/plan",
        json={"question": question}
    )
    plan_resp.raise_for_status()
    plan_data = plan_resp.json()["plan"]

    got_plan_route = plan_data["route"]
    got_plan_targets = plan_data["targets"]

    plan_route_ok = got_plan_route == case["expected_route"]
    plan_targets_ok = got_plan_targets == case["expected_targets"]
    plan_ok = plan_route_ok and plan_targets_ok

    if plan_ok:
        plan_pass += 1

    # -------------------------
    # API answer check
    # -------------------------
    ask_resp = requests.post(
        f"{BASE_URL}/ask_routed",
        json={"question": question}
    )
    ask_resp.raise_for_status()
    data = ask_resp.json()

    answer = data["answer"].strip()
    api_route = data.get("route")
    sources = data.get("sources", [])

    got_answer = not is_refusal(answer)
    answer_ok = got_answer == case["should_answer"]

    if answer_ok:
        answer_pass += 1

    api_route_ok = api_route == case["expected_route"]
    if api_route_ok:
        api_route_pass += 1

    # -------------------------
    # Source presence check
    # -------------------------
    if case["should_answer"]:
        sources_presence_ok = len(sources) > 0
    else:
        sources_presence_ok = len(sources) == 0

    if sources_presence_ok:
        sources_presence_pass += 1

    # -------------------------
    # Source document check
    # -------------------------
    expected_source_docs = set(case.get("expected_source_docs", []))
    returned_source_docs = {
        src["document_id"]
        for src in sources
        if "document_id" in src and src["document_id"] is not None
    }

    if expected_source_docs:
        source_docs_ok = expected_source_docs.issubset(returned_source_docs)
    else:
        source_docs_ok = len(returned_source_docs) == 0

    if source_docs_ok:
        source_docs_pass += 1

    print(f"\nCase {i}")
    print("Question           :", question)
    print("Expected plan       :", case["expected_route"], case["expected_targets"])
    print("Got plan            :", got_plan_route, got_plan_targets)
    print("Plan OK             :", plan_ok)
    print("API route           :", api_route)
    print("API route OK        :", api_route_ok)
    print("Answer              :", answer)
    print("Answer OK           :", answer_ok)
    print("Sources count       :", len(sources))
    print("Sources presence OK :", sources_presence_ok)
    print("Expected source docs:", sorted(expected_source_docs))
    print("Returned source docs:", sorted(returned_source_docs))
    print("Source docs OK      :", source_docs_ok)

print("\n--- Summary ---")
print(f"Planner                 : {plan_pass}/{len(cases)}")
print(f"API route               : {api_route_pass}/{len(cases)}")
print(f"Answer behavior         : {answer_pass}/{len(cases)}")
print(f"Sources presence        : {sources_presence_pass}/{len(cases)}")
print(f"Expected source docs    : {source_docs_pass}/{len(cases)}")