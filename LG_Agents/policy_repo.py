# policy_repo.py
import json, yaml, hashlib
from pathlib import Path
from typing import Optional, Dict
from policy_models import BundleRules, PolicyRef

class PolicyRepo:
    def __init__(self, base_dir: str = "rules"):
        self.base = Path(base_dir)

    def load_by_id(self, ruleset_id: str, version: str) -> tuple[BundleRules, PolicyRef]:
        path = next(self.base.glob(f"{ruleset_id}_{version}.yaml"), None) or \
               next(self.base.glob(f"{ruleset_id}.yaml"), None)
        if not path:
            raise FileNotFoundError(f"No policy file for {ruleset_id=} {version=}")
        data = yaml.safe_load(path.read_text())
        # hash the canonical JSON for auditing
        canon = json.dumps(data, sort_keys=True)
        h = hashlib.sha256(canon.encode()).hexdigest()[:12]
        rules = BundleRules.model_validate(data)
        ref = PolicyRef(ruleset_id=data.get("ruleset_id", ruleset_id),
                        version=data.get("version", version or "unknown"),
                        hash=h)
        return rules, ref

def resolve_rules(repo: PolicyRepo, pet, policy_ref: Optional[PolicyRef], overrides: Optional[dict]) -> tuple[BundleRules, PolicyRef]:
    # 1) choose a ruleset if none provided (segmenting by pet)
    if not policy_ref:
        ruleset_id = "puppy_dog" if (pet.species=="dog" and pet.age_months<12) else "adult_dog"
        version = "1.0.0"
        rules, ref = repo.load_by_id(ruleset_id, version)
    else:
        rules, ref = repo.load_by_id(policy_ref.ruleset_id, policy_ref.version)
    # 2) size-sensitive food swap (example of resolver logic)
    if pet.age_months < 12:
        core = []
        for sku in rules.core_items:
            if sku.startswith("FOO-PUP-"):
                sku = "FOO-PUP-LG-15" if pet.weight_lb >= 20 else "FOO-PUP-SM-5"
            core.append(sku)
        rules = rules.model_copy(update={"core_items": core})
    # 3) apply per-session overrides (only diffs)
    if overrides:
        rules = rules.model_copy(update=overrides)
    return rules, ref
