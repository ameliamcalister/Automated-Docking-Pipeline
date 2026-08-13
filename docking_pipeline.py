"""
Generalised AI-Guided Docking Pipeline
Works with any protein target, any PDB structure, any binding site

Usage:
    python docking_pipeline.py \
        --log compound_vina.log \
        --smiles "CCO..." \
        --receptor path/to/receptor.pdbqt \
        --center_x 45.0 --center_y 32.0 --center_z 18.0 \
        --size_x 25.0 --size_y 25.0 --size_z 25.0 \
        --target "EGFR kinase" \
        --pdb_id 1IEP \
        --cavity "ATP binding pocket, hinge region" \
        --key_residues "Thr790,Met793,Lys745,Glu762" \
        --chembl_id CHEMBL203 \
        --mode affinity \
        --chembl \
        --n 3 \
        --autodock \
        --yes

Output:
    <name>_report.txt — full report with AI suggestions, validation, and docking results
"""

import re
import argparse
import subprocess
import shutil
import tempfile
import json
import urllib.request
from datetime import datetime
from pathlib import Path

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("WARNING: RDKit not found. SMILES validation will be skipped.")

# ── Tool paths (update these for your system) ─────────────────────────────────
import os

VINA_PATH   = os.environ.get("VINA_PATH", "vina")
OBABEL_PATH = os.environ.get("OBABEL_PATH", "obabel")

# ── Iteration settings ────────────────────────────────────────────────────────
IMPROVEMENT_THRESHOLD = 0.3
PLATEAU_CYCLES        = 2
ADMET_LIMITS = {
    "MW":       (0,   620),
    "cLogP":    (-5,  6.0),
    "HBD":      (0,   3),
    "HBA":      (0,   10),
    "TPSA":     (0,   140),
    "RotBonds": (0,   12),
}


# ═══════════════════════════════════════════════════════════════════════════════
# GENERALISED SYSTEM PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_system_prompt(
    target_name: str,
    pdb_id: str,
    cavity_description: str,
    key_residues: list[str],
    grid_centre: tuple,
    grid_size: tuple,
    cavity_volume: str = "unknown",
    target_biology: str = "",
    reference_smiles: str = "",
    chembl_id: str = "",
) -> str:
    """
    Build a generalised system prompt for any protein target.
    All MRP1-specific content is replaced with user-provided parameters.
    """
    cx, cy, cz = grid_centre
    sx, sy, sz = grid_size

    residue_table = ""
    if key_residues:
        residue_table = (
            "Key residues in the binding site (from literature or manual inspection):\n"
            + "\n".join(f"  {r}" for r in key_residues)
        )
    else:
        residue_table = "Key residues: not specified — reason from auto-detected contacts."

    biology_section = target_biology if target_biology else (
        f"{target_name} is the target for this drug discovery campaign. "
        f"Design inhibitors that bind tightly to the specified binding site."
    )

    ref_section = ""
    if reference_smiles:
        ref_section = f"\nReference starting compound SMILES:\n  {reference_smiles}\n"

    chembl_section = ""
    if chembl_id:
        chembl_section = f"\nChEMBL target ID for reference data retrieval: {chembl_id}\n"

    return f"""
You are an expert computational medicinal chemist specialising in structure-based drug design.
You are working on inhibitor discovery against the following target:

─── TARGET INFORMATION ───────────────────────────────────────────────────────

Target name: {target_name}
PDB structure: {pdb_id}
{biology_section}
{chembl_section}
─── BINDING SITE GEOMETRY ────────────────────────────────────────────────────

Binding site description: {cavity_description}
Estimated cavity volume: {cavity_volume}

Docking grid (AutoDock Vina, {pdb_id}):
  centre_x = {cx} | centre_y = {cy} | centre_z = {cz}
  size_x   = {sx} | size_y   = {sy} | size_z   = {sz}
  grid spacing = 0.375 Å

{residue_table}
{ref_section}
─── SMILES VALIDITY RULES — MANDATORY ───────────────────────────────────────

Every SMILES you produce MUST be chemically valid.

1. NEVER use lowercase letters for novel fused heterocycles you construct from scratch.
   Write ALL bonds explicitly using uppercase atoms and = for double bonds.
   Example: quinoline written safely: C1=CC=CC2=NC=CC=C12

2. For well-known simple rings (benzene, pyridine, naphthalene) lowercase IS safe.

3. For ANY ring system with more than 2 fused rings where one is a heterocycle,
   write the heterocyclic ring(s) in Kekulé form even if carbocyclic rings use lowercase.

4. Verify ring closure numbers are always used in pairs and never reused.

5. If uncertain whether a SMILES is valid, choose a simpler modification instead.

If you cannot produce a valid SMILES, explicitly state:
"SMILES UNAVAILABLE — recommend redrawing in ChemDraw"
Never silently include a SMILES you are uncertain about.

─── OUTPUT FORMAT ────────────────────────────────────────────────────────────

For each suggested modification, always return:

1. Modification name (short, descriptive)
2. Change made (precise: "replace X with Y at position Z")
3. SMILES of the proposed compound (valid, canonical)
4. Estimated MW and cLogP change relative to parent
5. Residues newly targeted
6. Mechanistic rationale
7. ADMET flag (PASS / WARN + reason / FLAG + reason)
8. Synthetic accessibility note (1–2 sentences)

─── SCORING CONTEXT ──────────────────────────────────────────────────────────

AutoDock Vina scores are in kcal/mol (more negative = stronger predicted binding).
Do not treat differences < 0.3 kcal/mol as meaningful (Vina precision ~±0.5 kcal/mol).

─── WHAT YOU ARE NOT ─────────────────────────────────────────────────────────

Do not suggest modifications unrelated to the target biology above.
Do not provide cell-line IC50 predictions or clinical dosing extrapolations.
Keep responses focused and directly actionable for a computational docking workflow.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# CHEMBL: generalised target search + inhibitor fetch
# ═══════════════════════════════════════════════════════════════════════════════

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"


def search_chembl_target(query: str) -> str | None:
    """
    Search ChEMBL for a target by name or UniProt ID.
    Returns the ChEMBL target ID (e.g. 'CHEMBL203') or None if not found.
    """
    encoded = urllib.request.quote(query)
    url = f"{CHEMBL_API}/target/search.json?q={encoded}&limit=5"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        targets = data.get("targets", [])
        if targets:
            best = targets[0]
            print(f"  Found: {best.get('pref_name')} ({best.get('target_chembl_id')})")
            return best.get("target_chembl_id")
    except Exception as e:
        print(f"  ChEMBL target search failed: {e}")
    return None


def fetch_chembl_inhibitors(
    chembl_target_id: str,
    max_compounds: int = 10,
) -> list[dict]:
    """
    Fetch known inhibitors for any ChEMBL target ID.
    Returns list of dicts with smiles, activity_value, pchembl.
    """
    url = (
        f"{CHEMBL_API}/activity.json?"
        f"target_chembl_id={chembl_target_id}"
        f"&activity_type__in=IC50,Ki,Kd,EC50"
        f"&pchembl_value__isnull=false"
        f"&limit={max_compounds}"
        f"&order_by=-pchembl_value"
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"  ChEMBL fetch failed: {e}")
        return []

    results = []
    for act in data.get("activities", []):
        smiles   = act.get("canonical_smiles", "")
        pchembl  = act.get("pchembl_value")
        act_type = act.get("standard_type", "")
        name     = act.get("molecule_pref_name") or act.get("molecule_chembl_id", "")
        if smiles and pchembl:
            results.append({
                "smiles":        smiles,
                "name":          name,
                "activity_type": act_type,
                "pchembl":       float(pchembl),
                "ic50_nm":       round(10 ** (9 - float(pchembl)), 1),
            })
    return results


def format_chembl_context(compounds: list[dict], max_show: int = 5) -> str:
    """Format ChEMBL compounds as context for the prompt."""
    if not compounds:
        return ""
    lines = [
        f"\n─── CHEMBL REFERENCE DATA (top {min(len(compounds), max_show)} known inhibitors) ───",
        "Use these to identify pharmacophores correlating with high potency.",
        "",
    ]
    for i, c in enumerate(compounds[:max_show], 1):
        ic50_str = f"{c['ic50_nm']} nM" if c['ic50_nm'] else "unknown"
        lines.append(
            f"{i}. {c['name'] or 'Unknown'}\n"
            f"   SMILES: {c['smiles']}\n"
            f"   {c['activity_type']}: {ic50_str}  (pChEMBL={c['pchembl']:.2f})"
        )
    lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# VINA LOG PARSER
# ═══════════════════════════════════════════════════════════════════════════════

def parse_vina_log(log_path: str) -> dict:
    """Parse any AutoDock Vina log file."""
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")
    text = log_path.read_text()

    ligand        = re.search(r"Ligand:\s*(\S+)", text)
    receptor      = re.search(r"Rigid receptor:\s*(\S+)", text)
    exhaustiveness= re.search(r"Exhaustiveness:\s*(\d+)", text)
    seed          = re.search(r"random seed:\s*(-?\d+)", text)
    grid          = re.search(r"Grid center:\s*X\s*([\d.]+)\s*Y\s*([\d.]+)\s*Z\s*([\d.]+)", text)

    mode_pattern  = re.compile(r"^\s*(\d+)\s+([-\d.]+)\s+([\d.]+)\s+([\d.]+)", re.MULTILINE)
    modes = [
        {"mode": int(m.group(1)), "affinity": float(m.group(2)),
         "rmsd_lb": float(m.group(3)), "rmsd_ub": float(m.group(4))}
        for m in mode_pattern.finditer(text)
    ]
    if not modes:
        raise ValueError(f"No docking modes found in {log_path}.")

    return {
        "best_affinity":  modes[0]["affinity"],
        "all_modes":      modes,
        "ligand_name":    Path(ligand.group(1)).stem if ligand else "unknown",
        "receptor_name":  Path(receptor.group(1)).stem if receptor else "unknown",
        "grid_center":    (float(grid.group(1)), float(grid.group(2)), float(grid.group(3))) if grid else None,
        "exhaustiveness": int(exhaustiveness.group(1)) if exhaustiveness else None,
        "random_seed":    int(seed.group(1)) if seed else None,
        "n_modes":        len(modes),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RESIDUE CONTACT DETECTION (target-agnostic)
# ═══════════════════════════════════════════════════════════════════════════════

_AA3 = {
    "ALA":"Ala","ARG":"Arg","ASN":"Asn","ASP":"Asp","CYS":"Cys",
    "GLN":"Gln","GLU":"Glu","GLY":"Gly","HIS":"His","ILE":"Ile",
    "LEU":"Leu","LYS":"Lys","MET":"Met","PHE":"Phe","PRO":"Pro",
    "SER":"Ser","THR":"Thr","TRP":"Trp","TYR":"Tyr","VAL":"Val",
}


def parse_pdbqt_atoms(pdbqt_path: str) -> list[dict]:
    atoms = []
    with open(pdbqt_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            try:
                atoms.append({
                    "record":  line[0:6].strip(),
                    "resname": line[17:20].strip(),
                    "resnum":  int(line[22:26].strip()),
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                })
            except (ValueError, IndexError):
                continue
    return atoms


def detect_contacts(
    ligand_pdbqt: str,
    receptor_pdbqt: str,
    cutoff: float = 3.5,
    numbering_offset: int = 0,
) -> tuple[list[str], dict]:
    """
    Detect receptor residues within cutoff Å of any ligand atom.
    numbering_offset: add to PDBQT residue numbers to get canonical numbering.
    """
    ligand_atoms   = parse_pdbqt_atoms(ligand_pdbqt)
    receptor_atoms = parse_pdbqt_atoms(receptor_pdbqt)
    if not ligand_atoms:
        ligand_atoms = parse_pdbqt_atoms(ligand_pdbqt)

    contacts = {}
    for lat in ligand_atoms:
        for rat in receptor_atoms:
            dx = lat["x"] - rat["x"]
            dy = lat["y"] - rat["y"]
            dz = lat["z"] - rat["z"]
            dist = (dx*dx + dy*dy + dz*dz) ** 0.5
            if dist <= cutoff:
                key = (rat["resname"], rat["resnum"])
                if key not in contacts or dist < contacts[key]:
                    contacts[key] = dist

    residue_strings = []
    contact_detail  = {}
    for (resname, resnum), dist in sorted(contacts.items(), key=lambda x: x[1]):
        canonical_num = resnum + numbering_offset
        name = _AA3.get(resname.upper(), resname.capitalize())
        label = f"{name}{canonical_num}"
        residue_strings.append(label)
        contact_detail[label] = round(dist, 2)

    return residue_strings, contact_detail


def get_ligand_centre(ligand_pdbqt: str) -> tuple | None:
    atoms = parse_pdbqt_atoms(ligand_pdbqt)
    if not atoms:
        return None
    return (
        round(sum(a["x"] for a in atoms) / len(atoms), 3),
        round(sum(a["y"] for a in atoms) / len(atoms), 3),
        round(sum(a["z"] for a in atoms) / len(atoms), 3),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# RDKIT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_smiles(smiles: str) -> tuple[bool, object]:
    if not RDKIT_AVAILABLE:
        return True, None
    mol = Chem.MolFromSmiles(smiles.strip())
    return (True, mol) if mol else (False, None)


def calculate_properties(mol) -> dict:
    if mol is None:
        return {}
    return {
        "MW":       round(Descriptors.ExactMolWt(mol), 2),
        "cLogP":    round(Descriptors.MolLogP(mol), 2),
        "HBD":      rdMolDescriptors.CalcNumHBD(mol),
        "HBA":      rdMolDescriptors.CalcNumHBA(mol),
        "TPSA":     round(Descriptors.TPSA(mol), 1),
        "RotBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
    }


def check_admet(props: dict) -> list[str]:
    issues = []
    if not props:
        return issues
    mw, logp, hbd, hba, tpsa, rot = (
        props.get("MW", 0), props.get("cLogP", 0), props.get("HBD", 0),
        props.get("HBA", 0), props.get("TPSA", 0), props.get("RotBonds", 0)
    )
    if mw > 620:   issues.append(f"FLAG  MW = {mw} Da  (limit: ≤620)")
    elif mw > 600: issues.append(f"WARN  MW = {mw} Da  (soft limit: ≤600)")
    if logp > 6.0:   issues.append(f"FLAG  cLogP = {logp}  (limit: ≤6.0)")
    elif logp > 5.5: issues.append(f"WARN  cLogP = {logp}  (soft limit: ≤5.5)")
    if hbd > 3:    issues.append(f"FLAG  HBD = {hbd}  (limit: ≤3)")
    if hba > 10:   issues.append(f"FLAG  HBA = {hba}  (limit: ≤10)")
    if tpsa > 140: issues.append(f"FLAG  TPSA = {tpsa} Å²  (limit: ≤140)")
    if rot > 12:   issues.append(f"FLAG  RotBonds = {rot}  (limit: ≤12)")
    elif rot > 10: issues.append(f"WARN  RotBonds = {rot}  (soft limit: ≤10)")
    return issues


def extract_smiles_from_response(text: str) -> list[dict]:
    found = []
    for match in re.finditer(r"`([A-Za-z0-9@\[\]()=#\-+/\\%.,:]+)`", text):
        candidate = match.group(1)
        if any(c in candidate for c in "()=#[]@/\\%") and len(candidate) > 10:
            found.append({"smiles": candidate})
    for match in re.finditer(r"\*\*SMILES[^*]*\*\*[:\s`]*([A-Za-z0-9@\[\]()=#\-+/\\%.,:]+)", text):
        candidate = match.group(1).strip("`")
        if len(candidate) > 10 and candidate not in [f["smiles"] for f in found]:
            found.append({"smiles": candidate})
    return found


def validate_response_smiles(response_text: str) -> tuple[str, list[dict]]:
    if not RDKIT_AVAILABLE:
        return response_text, []
    candidates = extract_smiles_from_response(response_text)
    if not candidates:
        return response_text + "\n\n[RDKit] No SMILES detected.\n", []

    results = []
    lines   = ["", "=" * 60, "RDKIT VALIDATION RESULTS", "=" * 60]

    for i, item in enumerate(candidates, 1):
        smiles = item["smiles"]
        valid, mol = validate_smiles(smiles)
        if not valid:
            results.append({"smiles": smiles, "valid": False, "props": {}, "issues": []})
            lines += [f"\nSMILES {i}: INVALID ✗", f"  {smiles}",
                      "  RDKit cannot parse — do not dock.",
                      "  Fix: redraw in ChemDraw/MarvinSketch and re-export."]
            continue
        props  = calculate_properties(mol)
        issues = check_admet(props)
        results.append({"smiles": smiles, "valid": True, "props": props, "issues": issues})
        status = "PASS ✓" if not issues else ("FLAG ✗" if any("FLAG" in x for x in issues) else "WARN ⚠")
        lines += [f"\nSMILES {i}: {status}", f"  {smiles}",
                  f"  MW={props['MW']} Da  cLogP={props['cLogP']}  "
                  f"HBD={props['HBD']}  HBA={props['HBA']}  "
                  f"TPSA={props['TPSA']} Å²  RotBonds={props['RotBonds']}"]
        for issue in issues:
            lines.append(f"  ⚠  {issue}")

    n_valid   = sum(1 for r in results if r["valid"])
    n_invalid = len(results) - n_valid
    n_flagged = sum(1 for r in results if r["valid"] and any("FLAG" in x for x in r["issues"]))
    lines += ["", f"Summary: {n_valid}/{len(results)} valid | {n_flagged} ADMET flags | {n_invalid} invalid",
              "=" * 60]

    return response_text + "\n" + "\n".join(lines), results


# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-DOCKING
# ═══════════════════════════════════════════════════════════════════════════════

def find_tool(hardcoded_path: str, tool_name: str) -> str:
    if Path(hardcoded_path).exists():
        return hardcoded_path
    found = shutil.which(tool_name)
    if found:
        return found
    raise FileNotFoundError(f"Cannot find '{tool_name}'. Expected at {hardcoded_path}")


def smiles_to_pdbqt(smiles: str, name: str, work_dir: Path) -> Path | None:
    obabel   = find_tool(OBABEL_PATH, "obabel")
    smi_path  = work_dir / f"{name}.smi"
    sdf_path  = work_dir / f"{name}.sdf"
    pdbqt_path= work_dir / f"{name}.pdbqt"

    smi_path.write_text(smiles.strip())

    cmd_sdf = [obabel, str(smi_path), "-osdf", "-O", str(sdf_path),
               "--gen3d", "--forcefield", "mmff94", "-h", "--partialcharge", "gasteiger"]
    subprocess.run(cmd_sdf, capture_output=True, text=True)
    if not sdf_path.exists() or sdf_path.stat().st_size == 0:
        print(f"    ✗ obabel SDF conversion failed for {name}")
        return None

    cmd_pdbqt = [obabel, str(sdf_path), "-opdbqt", "-O", str(pdbqt_path),
                 "--partialcharge", "gasteiger"]
    subprocess.run(cmd_pdbqt, capture_output=True, text=True)
    if not pdbqt_path.exists() or pdbqt_path.stat().st_size == 0:
        print(f"    ✗ obabel PDBQT conversion failed for {name}")
        return None
    return pdbqt_path


def run_vina(
    pdbqt_path: Path, name: str, work_dir: Path,
    receptor: str, grid: dict, exhaustiveness: int = 8,
) -> tuple[float | None, Path | None]:
    vina      = find_tool(VINA_PATH, "vina")
    out_pdbqt = work_dir / f"{name}_docked.pdbqt"

    cmd = [
        vina,
        "--receptor",       receptor,
        "--ligand",         str(pdbqt_path),
        "--out",            str(out_pdbqt),
        "--center_x",       str(grid["center_x"]),
        "--center_y",       str(grid["center_y"]),
        "--center_z",       str(grid["center_z"]),
        "--size_x",         str(grid["size_x"]),
        "--size_y",         str(grid["size_y"]),
        "--size_z",         str(grid["size_z"]),
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes",      "9",
        "--energy_range",   "3",
    ]
    print(f"    Running Vina on {name}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    combined = result.stdout + result.stderr
    (work_dir / f"{name}_vina.log").write_text(combined)

    if not out_pdbqt.exists():
        print(f"    ✗ Vina failed: {result.stderr.strip()[:200]}")
        return None, None

    match = re.search(r"^\s*1\s+([-\d.]+)", combined, re.MULTILINE)
    affinity = float(match.group(1)) if match else None
    return affinity, out_pdbqt


def dock_all_valid(
    validation_results: list[dict], output_dir: Path,
    receptor: str, grid: dict, exhaustiveness: int = 8,
    dock_all: bool = False,
) -> list[dict]:
    docking_results = []
    for i, result in enumerate(validation_results, 1):
        if not result["valid"]:
            print(f"\n  Compound {i}: SKIPPED (invalid SMILES)")
            docking_results.append({**result, "docked": False, "reason": "invalid SMILES"})
            continue

        props     = result["props"]
        has_flag  = any("FLAG" in x for x in result["issues"])
        has_warn  = any("WARN" in x for x in result["issues"])
        status    = "FLAG ✗" if has_flag else ("WARN ⚠" if has_warn else "PASS ✓")
        print(f"\n  Compound {i}: {status}")
        print(f"  MW={props.get('MW','?')} Da  cLogP={props.get('cLogP','?')}")
        print(f"  SMILES: {result['smiles'][:80]}{'...' if len(result['smiles'])>80 else ''}")

        if not dock_all:
            while True:
                ans = input("  Dock this compound? [y/n]: ").strip().lower()
                if ans in ("y", "yes", "n", "no"):
                    break
            if ans in ("n", "no"):
                docking_results.append({**result, "docked": False, "reason": "skipped by user"})
                continue
        else:
            print("  Auto-docking...")

        name = f"suggestion_{i}"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path   = Path(tmp)
            pdbqt      = smiles_to_pdbqt(result["smiles"], name, tmp_path)
            if pdbqt is None:
                docking_results.append({**result, "docked": False, "reason": "obabel failed"})
                continue
            affinity, docked = run_vina(pdbqt, name, tmp_path, receptor, grid, exhaustiveness)
            if docked is None:
                docking_results.append({**result, "docked": False, "reason": "Vina failed"})
                continue
            final = output_dir / f"{name}_docked.pdbqt"
            shutil.copy(docked, final)
            log_src = tmp_path / f"{name}_vina.log"
            if log_src.exists():
                shutil.copy(log_src, output_dir / f"{name}_vina.log")
            if affinity:
                print(f"  ✓ Docked. Best affinity: {affinity:.2f} kcal/mol")
                print(f"  ✓ Saved: {final.name}")
            docking_results.append({
                **result, "docked": True, "affinity": affinity,
                "pdbqt_path": str(final), "compound_name": name,
            })
    return docking_results


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER (cavity-aware, generalised)
# ═══════════════════════════════════════════════════════════════════════════════

def build_prompt(
    smiles: str, docking_score: float, residues: list, contact_detail: dict,
    ligand_centre: tuple | None, grid_centre: tuple, grid_size: tuple,
    compound_name: str = "", notes: str = "", n_suggestions: int = 3,
    admet_mode: bool = False,
) -> str:
    name_line = f"Compound name: {compound_name}\n" if compound_name else ""
    cx, cy, cz = grid_centre
    sx, sy, sz = grid_size

    # Contact grading
    strong = {r: d for r, d in contact_detail.items() if d < 2.5}
    medium = {r: d for r, d in contact_detail.items() if 2.5 <= d < 3.0}
    weak   = {r: d for r, d in contact_detail.items() if d >= 3.0}

    contact_str = ""
    if strong:
        contact_str += "STRONG (<2.5 Å) — well-satisfied:\n" + "\n".join(f"  {r}: {d} Å" for r,d in strong.items()) + "\n\n"
    if medium:
        contact_str += "MEDIUM (2.5–3.0 Å):\n" + "\n".join(f"  {r}: {d} Å" for r,d in medium.items()) + "\n\n"
    if weak:
        contact_str += "WEAK (3.0–3.5 Å) — HIGH PRIORITY to strengthen:\n" + "\n".join(f"  {r}: {d} Å" for r,d in weak.items())
    if not contact_str:
        contact_str = "  Not detected — target full cavity"

    # Spatial analysis
    if ligand_centre:
        lx, ly, lz = ligand_centre
        ox = round(lx - cx, 2); oy = round(ly - cy, 2); oz = round(lz - cz, 2)
        dist = round((ox**2 + oy**2 + oz**2)**0.5, 2)
        primary = max([("X", abs(ox)), ("Y", abs(oy)), ("Z", abs(oz))], key=lambda x: x[1])
        dir_map = {"X": "negative X" if ox > 0 else "positive X",
                   "Y": "negative Y" if oy > 0 else "positive Y",
                   "Z": "negative Z" if oz > 0 else "positive Z"}
        spatial_str = (
            f"Ligand centre: ({lx}, {ly}, {lz})\n"
            f"Grid centre:   ({cx}, {cy}, {cz})\n"
            f"Displacement:  ΔX={ox} Å, ΔY={oy} Å, ΔZ={oz} Å (total {dist} Å)\n"
            f"Grid size:     {sx} × {sy} × {sz} Å\n"
            f"Expand toward: {dir_map[primary[0]]} ({primary[1]:.2f} Å available)"
        )
    else:
        spatial_str = f"Grid centre: ({cx}, {cy}, {cz}) | Size: {sx} × {sy} × {sz} Å"

    notes_section = f"\nAdditional observations: {notes}" if notes else ""

    if admet_mode:
        task = f"""Affinity optimisation has plateaued. Switch to ADMET optimisation mode.
Suggest {n_suggestions} modifications that reduce cLogP and improve drug-likeness
while maintaining affinity as close to {docking_score:.2f} kcal/mol as possible.
Use polar bioisosteres, HBD/HBA additions, or solubilising groups."""
    else:
        task = f"""Suggest {n_suggestions} structural modifications to improve binding affinity.
Use the spatial and contact data above to reason about unfilled binding site space.
Focus on weak contacts and directions with available cavity space.
For each suggestion state which strategy it uses (ChEMBL fragment / uncontacted residue / pose-guided)."""

    return f"""Compound docked into binding site.

{name_line}SMILES: {smiles}
Best Vina score: {docking_score:.2f} kcal/mol

SPATIAL ANALYSIS:
{spatial_str}

RESIDUE CONTACTS (auto-detected, ≤3.5 Å):
{contact_str}
{notes_section}

{task}"""


# ═══════════════════════════════════════════════════════════════════════════════
# ITERATION HISTORY
# ═══════════════════════════════════════════════════════════════════════════════

def load_history(history_file: str) -> list[float]:
    p = Path(history_file)
    return json.loads(p.read_text()) if p.exists() else []

def save_history(history_file: str, history: list[float]) -> None:
    Path(history_file).write_text(json.dumps(history))

def resolve_mode(manual_mode: str, history: list[float], target: float) -> tuple[str, str]:
    if manual_mode == "affinity":
        best = max(abs(x) for x in history) if history else 0
        return "affinity", f"Manual: AFFINITY. Best so far: {best:.2f}. Target: {target}."
    if manual_mode == "admet":
        return "admet", "Manual: ADMET optimisation."
    if not history:
        return "affinity", "Auto: starting affinity optimisation."
    best = max(abs(x) for x in history)
    if best >= target:
        return "admet", f"Target {target} reached ({best:.2f}). Switching to ADMET."
    if len(history) >= PLATEAU_CYCLES + 1:
        recent = [abs(x) for x in history[-(PLATEAU_CYCLES+1):]]
        improvements = [recent[i+1]-recent[i] for i in range(len(recent)-1)]
        avg = sum(improvements)/len(improvements)
        if avg < IMPROVEMENT_THRESHOLD:
            return "admet", f"Plateau detected (avg Δ={avg:.2f} kcal/mol). Switching to ADMET."
    return "affinity", f"Auto: cycle {len(history)}, best {best:.2f} kcal/mol."


# ═══════════════════════════════════════════════════════════════════════════════
# LLM BACKEND
# ═══════════════════════════════════════════════════════════════════════════════

OLLAMA_URL    = "http://127.0.0.1:11434/api/chat"
DEFAULT_LOCAL = "qwen2.5:1.5b"
DEFAULT_CLOUD = "claude-sonnet-4-6"

def call_llm(prompt: str, system: str, backend: str = "claude", model: str = "") -> str:
    if backend == "ollama":
        return _call_ollama(prompt, system, model or DEFAULT_LOCAL)
    return _call_claude(prompt, system, model or DEFAULT_CLOUD)

def _call_claude(prompt: str, system: str, model: str) -> str:
    if not ANTHROPIC_AVAILABLE:
        raise ImportError("anthropic package not installed. Run: pip install anthropic")
    client   = anthropic.Anthropic()
    response = client.messages.create(
        model=model, max_tokens=1500, system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text

def _call_ollama(prompt: str, system: str, model: str) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 1500},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())["message"]["content"]


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT WRITER
# ═══════════════════════════════════════════════════════════════════════════════

def format_docking_section(docking_results: list[dict]) -> str:
    if not docking_results:
        return "No docking performed."
    lines = []
    for i, r in enumerate(docking_results, 1):
        lines.append(f"Compound {i}")
        lines.append(f"  SMILES: {r['smiles']}")
        if not r["valid"]:
            lines.append("  Status: INVALID")
        elif not r["docked"]:
            lines.append(f"  Status: Not docked ({r.get('reason','')})")
        else:
            aff = r.get("affinity")
            lines.append(f"  Status: Docked ✓")
            lines.append(f"  Best affinity: {aff:.2f} kcal/mol" if aff else "  Best affinity: see log")
            lines.append(f"  PyMOL file: {Path(r['pdbqt_path']).name}")
        lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generalised AI-guided docking pipeline — works with any protein target"
    )
    # Required inputs
    parser.add_argument("--log",         required=True,  help="Vina log .txt file")
    parser.add_argument("--smiles",      required=True,  help="SMILES of docked compound")
    parser.add_argument("--receptor",    required=True,  help="Receptor PDBQT file path")
    parser.add_argument("--center_x",   type=float, required=True, help="Grid centre X")
    parser.add_argument("--center_y",   type=float, required=True, help="Grid centre Y")
    parser.add_argument("--center_z",   type=float, required=True, help="Grid centre Z")
    parser.add_argument("--size_x",     type=float, required=True, help="Grid size X (Å)")
    parser.add_argument("--size_y",     type=float, required=True, help="Grid size Y (Å)")
    parser.add_argument("--size_z",     type=float, required=True, help="Grid size Z (Å)")
    # Target information
    parser.add_argument("--target",     default="Unknown target", help="Target protein name")
    parser.add_argument("--pdb_id",     default="unknown",        help="PDB ID e.g. 5UJA")
    parser.add_argument("--cavity",     default="",               help="Binding site description")
    parser.add_argument("--key_residues",nargs="*", default=[],   help="Key binding residues")
    parser.add_argument("--biology",    default="",               help="Target biology description")
    parser.add_argument("--chembl_id",  default="",               help="ChEMBL target ID e.g. CHEMBL4523")
    parser.add_argument("--chembl_search", default="",            help="Search ChEMBL by name/UniProt")
    parser.add_argument("--offset",     type=int, default=0,      help="Residue numbering offset")
    # Run options
    parser.add_argument("--name",       default="",     help="Compound name/ID")
    parser.add_argument("--pdbqt",      default="",     help="Docked ligand PDBQT for contact detection")
    parser.add_argument("--residues",   nargs="*", default=[], help="Manual residue override")
    parser.add_argument("--notes",      default="",     help="Notes for AI")
    parser.add_argument("--n",          type=int, default=3, help="Number of suggestions")
    parser.add_argument("--mode",       default="auto", choices=["affinity","admet","auto"])
    parser.add_argument("--target_affinity", type=float, default=13.0, help="Target ΔG (abs)")
    parser.add_argument("--output",     default="",     help="Report filename")
    parser.add_argument("--history",    default="",     help="Comma-separated past affinities")
    parser.add_argument("--history-file", default="docking_history.json")
    parser.add_argument("--autodock",   action="store_true")
    parser.add_argument("--yes",        action="store_true", help="Dock all without asking")
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--backend",    default="claude", choices=["claude","ollama"])
    parser.add_argument("--model",      default="")
    parser.add_argument("--chembl",     action="store_true", help="Fetch ChEMBL reference data")

    args   = parser.parse_args()
    name   = args.name
    grid   = {"center_x": args.center_x, "center_y": args.center_y, "center_z": args.center_z,
              "size_x": args.size_x, "size_y": args.size_y, "size_z": args.size_z}
    grid_centre = (args.center_x, args.center_y, args.center_z)
    grid_size   = (args.size_x, args.size_y, args.size_z)

    # ── Step 1: Parse log ──
    print(f"Parsing Vina log: {args.log}")
    log_data = parse_vina_log(args.log)
    name     = name or log_data["ligand_name"]
    print(f"  Ligand: {log_data['ligand_name']} | Best affinity: {log_data['best_affinity']:.2f} kcal/mol")

    # ── Step 1b: History ──
    history = [abs(float(x)) for x in args.history.split(",") if x.strip()] if args.history else load_history(args.history_file)
    history.append(abs(log_data["best_affinity"]))
    save_history(args.history_file, history)
    mode, iter_msg = resolve_mode(args.mode, history, args.target_affinity)
    print(f"  Mode: {mode.upper()} — {iter_msg}")

    # ── Step 1c: ChEMBL ──
    chembl_context = ""
    chembl_id = args.chembl_id
    if args.chembl:
        if args.chembl_search and not chembl_id:
            print(f"\nSearching ChEMBL for: {args.chembl_search}")
            chembl_id = search_chembl_target(args.chembl_search) or ""
        if chembl_id:
            print(f"\nFetching inhibitors for {chembl_id}...")
            compounds = fetch_chembl_inhibitors(chembl_id)
            if compounds:
                print(f"  ✓ {len(compounds)} compounds (best pChEMBL={compounds[0]['pchembl']:.2f})")
                chembl_context = format_chembl_context(compounds)
        elif args.chembl:
            print("  ⚠  No ChEMBL target ID provided. Use --chembl_id or --chembl_search.")

    # ── Step 2: Residue contacts ──
    residues, contact_detail, ligand_centre = list(args.residues), {}, None
    if args.pdbqt and not residues:
        if Path(args.pdbqt).exists():
            print(f"\nDetecting contacts from: {args.pdbqt}")
            residues, contact_detail = detect_contacts(args.pdbqt, args.receptor, numbering_offset=args.offset)
            ligand_centre = get_ligand_centre(args.pdbqt)
            print(f"  ✓ {len(residues)} residues within 3.5 Å")
            if ligand_centre:
                dist = round(sum((a-b)**2 for a,b in zip(ligand_centre, grid_centre))**0.5, 2)
                print(f"  ✓ Ligand centre: {ligand_centre} ({dist} Å from grid centre)")

    # ── Step 3: Build system prompt ──
    system_prompt = build_system_prompt(
        target_name      = args.target,
        pdb_id           = args.pdb_id,
        cavity_description = args.cavity or "Binding site defined by docking grid",
        key_residues     = args.key_residues,
        grid_centre      = grid_centre,
        grid_size        = grid_size,
        target_biology   = args.biology,
        chembl_id        = chembl_id,
    )

    # ── Step 4: Build prompt + call LLM ──
    full_notes = (args.notes + "\n" + chembl_context).strip() if chembl_context else args.notes
    prompt = build_prompt(
        smiles=args.smiles, docking_score=log_data["best_affinity"],
        residues=residues, contact_detail=contact_detail,
        ligand_centre=ligand_centre, grid_centre=grid_centre, grid_size=grid_size,
        compound_name=name, notes=full_notes, n_suggestions=args.n,
        admet_mode=(mode == "admet"),
    )

    print(f"\nSending to {args.backend.upper()} ({args.n} suggestions, mode={mode})...")
    suggestions = call_llm(prompt, system_prompt, args.backend, args.model)

    # ── Step 5: Validate ──
    suggestions_annotated, validation_results = validate_response_smiles(suggestions)
    if RDKIT_AVAILABLE:
        n_inv  = sum(1 for r in validation_results if not r["valid"])
        n_flag = sum(1 for r in validation_results if r["valid"] and any("FLAG" in x for x in r["issues"]))
        if n_inv:   print(f"  ⚠  {n_inv} invalid SMILES")
        if n_flag:  print(f"  ⚠  {n_flag} ADMET flags")
        if not n_inv and not n_flag: print("  ✓  All SMILES valid")

    # ── Step 6: Auto-dock ──
    docking_results = []
    if args.autodock and validation_results:
        output_dir = Path(args.output).parent if args.output else Path(".")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nAUTO-DOCKING — {args.target} ({args.pdb_id})")
        print(f"Receptor: {args.receptor}")
        docking_results = dock_all_valid(
            validation_results, output_dir, args.receptor, grid,
            args.exhaustiveness, dock_all=args.yes,
        )
        n_docked = sum(1 for r in docking_results if r.get("docked"))
        print(f"\n  {n_docked}/{len(docking_results)} docked.")

    # ── Step 7: Print + save ──
    print("\n" + "="*60 + "\nSUGGESTIONS\n" + "="*60)
    print(suggestions_annotated)

    output_path = args.output or f"{name}_report.txt"
    report = (
        f"DOCKING PIPELINE REPORT\nTarget: {args.target} ({args.pdb_id})\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n{'='*60}\n\n"
        f"COMPOUND: {name}\nSMILES: {args.smiles}\n\n"
        f"Best affinity: {log_data['best_affinity']:.2f} kcal/mol\n"
        f"Mode: {mode.upper()} — {iter_msg}\n\n"
        f"{'='*60}\nSUGGESTIONS\n{'='*60}\n\n{suggestions_annotated}\n"
    )
    if docking_results:
        report += f"\n{'='*60}\nDOCKING RESULTS\n{'='*60}\n\n{format_docking_section(docking_results)}"
    Path(output_path).write_text(report)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
