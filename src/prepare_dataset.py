import json
from pathlib import Path
from datasets import Dataset


REQUIRED_KEYS = {"prompt", "chosen", "rejected"}
DATA_PATH = Path(__file__).parent.parent / "data" / "hhh_dataset.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Linha {line_num} inválida: {e}") from e
            records.append(record)
    return records


def validate_records(records: list[dict]) -> None:
    for i, record in enumerate(records, start=1):
        missing = REQUIRED_KEYS - record.keys()
        if missing:
            raise KeyError(
                f"Registro {i} está faltando as chaves: {missing}"
            )
    print(f"[OK] {len(records)} registros validados com sucesso.")


def build_hf_dataset(records: list[dict]) -> Dataset:
    data = {key: [r[key] for r in records] for key in REQUIRED_KEYS}
    dataset = Dataset.from_dict(data)
    return dataset


def prepare(path: Path = DATA_PATH) -> Dataset:
    print(f"Carregando dataset de: {path}")
    records = load_jsonl(path)
    validate_records(records)
    dataset = build_hf_dataset(records)
    print(f"Dataset criado: {dataset}")
    return dataset


if __name__ == "__main__":
    ds = prepare()
    print("\nExemplo de entrada:")
    print(f"  prompt  : {ds[0]['prompt']}")
    print(f"  chosen  : {ds[0]['chosen']}")
    print(f"  rejected: {ds[0]['rejected']}")