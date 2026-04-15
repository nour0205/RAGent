def build_study_hint(route, sources):
    if not sources:
        return None

    combined_text = " ".join([s.text.lower() for s in sources])

    if route == "concept_explanation":
        if "normal form" in combined_text:
            return "Focus on 1NF, 2NF, and 3NF, and how each one reduces redundancy and anomalies."
        if "overfitting" in combined_text:
            return "Make sure you understand overfitting vs underfitting and how they relate to training and test performance."
        return "Review the main definition first, then the related ideas and examples."

    elif route == "source_recall":
        doc_ids = list({s.document_id for s in sources})
        return f"Go back to {', '.join(doc_ids)} and reread the section where this concept is introduced."

    elif route == "exam_preparation":
        hints = []

        if "anomal" in combined_text:
            hints.append("start with update, insertion, and deletion anomalies")

        if "normal form" in combined_text or "1nf" in combined_text or "2nf" in combined_text or "3nf" in combined_text:
            hints.append("then review the normal forms (1NF, 2NF, 3NF)")

        if "transitive" in combined_text or "dependency" in combined_text:
            hints.append("make sure you understand transitive dependency")

        if "decomposition" in combined_text:
            hints.append("finish with decomposition and why it fixes bad table design")

        if "overfitting" in combined_text or "underfitting" in combined_text:
            hints.append("start with overfitting vs underfitting")

        if "bias" in combined_text and "variance" in combined_text:
            hints.append("then review the bias-variance tradeoff")

        if "cross-validation" in combined_text:
            hints.append("finally understand how cross-validation helps detect overfitting")

        if hints:
            return "Study priority: " + " → ".join(hints) + "."
        return "Focus on the most important concepts first, then review supporting techniques and examples."

    else:
        doc_ids = list({s.document_id for s in sources})
        return f"Review the relevant material in: {', '.join(doc_ids)}"