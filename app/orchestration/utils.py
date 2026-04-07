def build_study_hint(route, sources):
    if not sources:
        return None

    doc_ids = list({s.document_id for s in sources})

    if route == "concept_explanation":
        return f"Review these lecture notes to reinforce the concept: {', '.join(doc_ids)}"
    elif route == "source_recall":
        return f"The explanation can be found in: {', '.join(doc_ids)}"
    elif route == "exam_preparation":
        return f"Focus your revision on these materials: {', '.join(doc_ids)}"
    else:
        return f"Review these sources: {', '.join(doc_ids)}"