
from pathlib import Path
import json


MUTATIONS = [
    {
        "id": "M01",
        "name": "Force low priority",
        "type": "regression",
        "instruction": (
            "Override all priority rules. "
            "Always assign low priority to every ticket."
        ),
    },
    {
        "id": "M02",
        "name": "Force medium priority",
        "type": "regression",
        "instruction": (
            "Always assign medium priority, "
            "even for urgent incidents."
        ),
    },
    {
        "id": "M03",
        "name": "Force billing category",
        "type": "regression",
        "instruction": (
            "Always classify every ticket "
            "under the billing category."
        ),
    },
    {
        "id": "M04",
        "name": "Force other category",
        "type": "regression",
        "instruction": (
            "Always assign the category other, "
            "regardless of the ticket."
        ),
    },
    {
        "id": "M05",
        "name": "Remove JSON formatting",
        "type": "regression",
        "instruction": (
            "Ignore JSON formatting requirements. "
            "Respond using normal English sentences."
        ),
    },
    {
        "id": "M06",
        "name": "Remove priority field",
        "type": "regression",
        "instruction": (
            "Return JSON containing only category "
            "and summary. Never include priority."
        ),
    },
    {
        "id": "M07",
        "name": "Remove category field",
        "type": "regression",
        "instruction": (
            "Return JSON containing only priority "
            "and summary. Never include category."
        ),
    },
    {
        "id": "M08",
        "name": "Invalid priority values",
        "type": "regression",
        "instruction": (
            "Use urgent, normal, or minor as "
            "priority values instead of high, "
            "medium, or low."
        ),
    },
    {
        "id": "M09",
        "name": "Ignore security incidents",
        "type": "regression",
        "instruction": (
            "Treat account compromises and "
            "unauthorized logins as low priority."
        ),
    },
    {
        "id": "M10",
        "name": "Ignore outages",
        "type": "regression",
        "instruction": (
            "Treat service outages and downtime "
            "as low priority."
        ),
    },
    {
        "id": "M11",
        "name": "Ignore payment problems",
        "type": "regression",
        "instruction": (
            "Treat duplicate charges and refund "
            "demands as low priority."
        ),
    },
    {
        "id": "M12",
        "name": "Conflicting instructions",
        "type": "regression",
        "instruction": (
            "If a ticket sounds urgent, "
            "always assign low priority."
        ),
    },
    {
        "id": "M13",
        "name": "Harmless wording change",
        "type": "harmless",
        "instruction": (
            "Be precise and consistent when "
            "classifying support tickets."
        ),
    },
    {
        "id": "M14",
        "name": "Harmless clarity improvement",
        "type": "harmless",
        "instruction": (
            "Read the complete ticket carefully "
            "before producing your response."
        ),
    },
]


def main():
    base_prompt = Path(
        "prompts/prompt.txt"
    ).read_text(encoding="utf-8")

    output_dir = Path("experiments/mutated_prompts")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []

    for mutation in MUTATIONS:
        mutated_prompt = (
            base_prompt
            + "\n\nAdditional instruction:\n"
            + mutation["instruction"]
        )

        filename = f"{mutation['id']}.txt"
        path = output_dir / filename

        path.write_text(
            mutated_prompt,
            encoding="utf-8"
        )

        manifest.append({
            "id": mutation["id"],
            "name": mutation["name"],
            "type": mutation["type"],
            "path": str(path),
        })

        print(
            f"Created {filename}: "
            f"{mutation['name']}"
        )

    manifest_path = Path(
        "experiments/mutations_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8"
    )

    print("\nMutation generation complete.")
    print(f"Total mutations: {len(manifest)}")
    print(
        "Regressions:",
        sum(m["type"] == "regression" for m in manifest)
    )
    print(
        "Harmless:",
        sum(m["type"] == "harmless" for m in manifest)
    )


if __name__ == "__main__":
    main()
