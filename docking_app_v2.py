"""
Generalised AI-Guided Docking Pipeline v2 — Streamlit Interface
New features:
  - Autonomous iterative loop (runs N cycles automatically)
  - In-browser 3D pose visualisation using 3Dmol.js
  - Fixed random seed option for reproducibility

Run with:
    streamlit run docking_app_v2.py

Requires docking_pipeline.py in the same folder.
"""

import streamlit as st
import sys
import os
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="AI Docking Pipeline",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');
:root {
    --bg: #0d0f14; --surface: #13161e; --border: #1e2330;
    --accent: #00d4aa; --accent2: #7c3aed; --warn: #f59e0b;
    --danger: #ef4444; --text: #e2e8f0; --muted: #64748b;
}
.stApp { background: var(--bg); color: var(--text); font-family: 'Syne', sans-serif; }
h1,h2,h3 { font-family: 'Syne', sans-serif; font-weight: 800; }
.main-title {
    font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em;
    background: linear-gradient(135deg, #00d4aa, #7c3aed);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.subtitle { color: var(--muted); font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; margin-bottom: 1.5rem; }
.stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1rem 1.3rem; margin-bottom: 0.6rem; }
.stat-value { font-size: 1.8rem; font-weight: 800; color: var(--accent); font-family: 'JetBrains Mono', monospace; }
.stat-label { color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
.result-block { background: var(--surface); border: 1px solid var(--border); border-left: 3px solid var(--accent); border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.5rem; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; }
.result-block.warn { border-left-color: var(--warn); }
.result-block.error { border-left-color: var(--danger); }
.badge { display: inline-block; padding: 0.12rem 0.5rem; border-radius: 20px; font-size: 0.68rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; text-transform: uppercase; }
.badge-pass { background: #064e3b; color: #34d399; }
.badge-warn { background: #451a03; color: #fbbf24; }
.badge-flag { background: #450a0a; color: #f87171; }
.badge-invalid { background: #1e1b4b; color: #a5b4fc; }
.affinity-big { font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 700; color: var(--accent); }
.divider { border: none; border-top: 1px solid var(--border); margin: 1.2rem 0; }
.cycle-header { background: linear-gradient(135deg,#1e2330,#13161e); border: 1px solid #2C5F8A; border-radius: 10px; padding: 0.8rem 1.2rem; margin: 1rem 0 0.5rem; }
.stButton > button { background: linear-gradient(135deg, #00d4aa, #059669) !important; color: #0d0f14 !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; border: none !important; border-radius: 8px !important; width: 100%; }
</style>
""", unsafe_allow_html=True)

# ── Import pipeline ───────────────────────────────────────────────────────────
pipeline_dir = Path(__file__).parent
sys.path.insert(0, str(pipeline_dir))

try:
    from docking_pipeline import (
        parse_vina_log, detect_contacts, get_ligand_centre,
        search_chembl_target, fetch_chembl_inhibitors, format_chembl_context,
        build_system_prompt, build_prompt, validate_response_smiles,
        smiles_to_pdbqt, run_vina, format_docking_section,
        resolve_mode, call_llm, RDKIT_AVAILABLE, VINA_PATH, OBABEL_PATH,
    )
    import anthropic
    PIPELINE_LOADED = True
except ImportError as e:
    PIPELINE_LOADED = False
    PIPELINE_ERROR  = str(e)

# ═══════════════════════════════════════════════════════════════════════════════
# 3DMOL VIEWER HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def render_3dmol(pdbqt_content: str, receptor_content: str = "", height: int = 480) -> str:
    """
    Enhanced 3Dmol viewer with:
    - Receptor shown as cartoon + transparent surface for context
    - Brightness/opacity slider for receptor
    - Ligand shown as balls-and-sticks coloured by element
    - Residues near ligand highlighted
    - Controls: Reset, Surface, Spin, receptor opacity slider
    """
    import json as _json

    # Extract first MODEL block (best pose)
    lines = pdbqt_content.split("\n")
    model_lines, in_model = [], False
    for line in lines:
        if line.startswith("MODEL"):
            in_model = True; model_lines = []; continue
        if line.startswith("ENDMDL"): break
        if in_model and line.startswith(("ATOM","HETATM","CONECT")):
            model_lines.append(line)
    if not model_lines:
        model_lines = [l for l in lines if l.startswith(("ATOM","HETATM"))]
    ligand_pdb = "\n".join(model_lines)

    # Get ligand centre for nearby residue highlighting
    lig_coords = []
    for line in model_lines:
        if line.startswith(("ATOM","HETATM")):
            try:
                lig_coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            except: pass
    if lig_coords:
        lx = sum(c[0] for c in lig_coords)/len(lig_coords)
        ly = sum(c[1] for c in lig_coords)/len(lig_coords)
        lz = sum(c[2] for c in lig_coords)/len(lig_coords)
    else:
        lx, ly, lz = 0, 0, 0

    # Receptor: sample every 4th atom for full protein coverage
    # This gives ~3000 atoms from 12000, covering the entire structure
    rec_all = [l for l in receptor_content.split("\n") if l.startswith(("ATOM","HETATM"))]
    # Try CA atoms first using strict column check
    rec_ca_lines = [l for l in rec_all if len(l) > 15 and l[12:16].strip() in ("CA", "C4'")]
    # If CA detection gives good coverage use it, otherwise sample every 4th atom
    if len(rec_ca_lines) >= 100:
        receptor_pdb = "\n".join(rec_ca_lines[:2000])
    else:
        # Sample every 4th atom — gives full spatial coverage of entire protein
        sampled = rec_all[::4][:2500]
        receptor_pdb = "\n".join(sampled)

    # All receptor atoms near ligand (within 8 Å) for binding site highlight
    nearby_lines = []
    for line in rec_all:
        try:
            rx,ry,rz = float(line[30:38]),float(line[38:46]),float(line[46:54])
            if ((rx-lx)**2+(ry-ly)**2+(rz-lz)**2)**0.5 < 8.0:
                nearby_lines.append(line)
        except: pass
    nearby_pdb = "\n".join(nearby_lines[:300])

    ligand_json  = _json.dumps(ligand_pdb)
    receptor_json= _json.dumps(receptor_pdb)
    nearby_json  = _json.dumps(nearby_pdb)
    has_receptor = bool(rec_all)

    html = f"""<!DOCTYPE html>
<html>
<head>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:#0d0f14; font-family:monospace; }}
  #viewer {{ width:100%; height:{height}px; position:relative; }}
  .top-bar {{ position:absolute; top:0; left:0; right:0; z-index:100;
              display:flex; align-items:center; gap:6px; padding:8px;
              background:linear-gradient(#0d0f14cc,transparent); }}
  .badge {{ background:#064e3b; color:#34d399; font-size:11px;
            padding:3px 8px; border-radius:4px; white-space:nowrap; }}
  .ctrl-btn {{ background:#13161e; border:1px solid #2e3748; color:#e2e8f0;
               padding:4px 10px; border-radius:5px; cursor:pointer; font-size:11px; }}
  .ctrl-btn:hover {{ background:#1e2330; }}
  .ctrl-btn.active {{ background:#064e3b; border-color:#34d399; color:#34d399; }}
  .slider-wrap {{ display:flex; align-items:center; gap:5px; margin-left:auto; }}
  .slider-label {{ color:#64748b; font-size:10px; white-space:nowrap; }}
  #opSlider {{ width:80px; accent-color:#00d4aa; }}
  .info {{ position:absolute; bottom:8px; left:8px; color:#475569;
           font-size:10px; z-index:100; }}
  .legend {{ position:absolute; bottom:8px; right:8px; z-index:100;
             display:flex; flex-direction:column; gap:3px; align-items:flex-end; }}
  .leg-item {{ display:flex; align-items:center; gap:5px; font-size:10px; color:#94a3b8; }}
  .leg-dot {{ width:10px; height:10px; border-radius:50%; }}
</style>
</head>
<body>
<div id="viewer">
  <div class="top-bar">
    <div class="badge" id="status">Loading...</div>
    <button class="ctrl-btn" onclick="resetView()">⟳ Reset</button>
    <button class="ctrl-btn" id="surfBtn" onclick="toggleSurface()">◈ Surface</button>
    <button class="ctrl-btn" id="spinBtn" onclick="toggleSpin()">↻ Spin</button>
    <button class="ctrl-btn" id="styleBtn" onclick="toggleLigStyle()">⬡ Style</button>
    <div class="slider-wrap">
      <span class="slider-label">Receptor opacity</span>
      <input type="range" id="opSlider" min="0" max="100" value="55"
             oninput="setReceptorOpacity(this.value/100)">
      <span class="slider-label" id="opVal">55%</span>
    </div>
  </div>
  <div class="info">Scroll to zoom &nbsp;·&nbsp; Drag to rotate &nbsp;·&nbsp; Right-drag to pan</div>
  <div class="legend">
    <div class="leg-item"><div class="leg-dot" style="background:#4A90D9"></div>Receptor</div>
    <div class="leg-item"><div class="leg-dot" style="background:#00d4aa"></div>Binding site</div>
    <div class="leg-item"><div class="leg-dot" style="background:#FFD700"></div>Ligand</div>
  </div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.3/3Dmol-min.js"></script>
<script>
var ligandData  = {ligand_json};
var recData     = {receptor_json};
var nearbyData  = {nearby_json};
var hasRec      = {'true' if has_receptor else 'false'};

var viewer     = null;
var surfaceOn  = false;
var spinning   = false;
var ligStyle   = 0;
var recOpacity = 0.55;

function initViewer() {{
  viewer = $3Dmol.createViewer(document.getElementById('viewer'), {{
    backgroundColor: '#0d0f14', antialias: true,
  }});
  loadMolecules();
}}

function loadMolecules() {{
  try {{
    var ligIdx = 0;

    if (hasRec && recData.trim().length > 10) {{
      // Full receptor as cartoon
      viewer.addModel(recData, 'pdb');
      viewer.setStyle({{model:0}}, {{
        cartoon: {{ color:'spectrum', opacity: recOpacity, thickness:0.4 }}
      }});
      ligIdx = 1;

      // Nearby residues highlighted differently
      if (nearbyData.trim().length > 10) {{
        viewer.addModel(nearbyData, 'pdb');
        viewer.setStyle({{model:2}}, {{
          cartoon: {{ color:'#4A90D9', opacity:0.85, thickness:0.5 }},
        }});
        viewer.addStyle({{model:2}}, {{
          stick: {{ color:'#4A90D9', radius:0.06, opacity:0.5 }}
        }});
      }}
    }}

    // Ligand
    viewer.addModel(ligandData, 'pdb');
    setLigandStyle(ligIdx, 0);

    viewer.zoomTo({{model:ligIdx}});
    viewer.render();

    var nAtoms = (ligandData.match(/^(?:ATOM|HETATM)/mg)||[]).length;
    var nRecRes = (recData.match(/^(?:ATOM|HETATM)/mg)||[]).length;
    document.getElementById('status').textContent =
      nAtoms + ' ligand atoms · ' +
      (hasRec ? nRecRes + ' receptor residues (Cα)' : 'no receptor');
    document.getElementById('status').style.background = '#064e3b';

    window._ligIdx = ligIdx;

  }} catch(e) {{
    document.getElementById('status').textContent = 'Error: ' + e.message;
    document.getElementById('status').style.background = '#450a0a';
    document.getElementById('status').style.color = '#f87171';
  }}
}}

function setLigandStyle(idx, style) {{
  viewer.setStyle({{model:idx}}, {{}});
  if (style === 0) {{
    viewer.setStyle({{model:idx}}, {{
      stick: {{ colorscheme:'elementWithCarbon', radius:0.15 }}
    }});
    viewer.addStyle({{model:idx}}, {{
      sphere: {{ colorscheme:'elementWithCarbon', radius:0.28 }}
    }});
  }} else if (style === 1) {{
    viewer.setStyle({{model:idx}}, {{
      stick: {{ colorscheme:'elementWithCarbon', radius:0.20 }}
    }});
  }} else {{
    viewer.setStyle({{model:idx}}, {{
      sphere: {{ colorscheme:'elementWithCarbon', radius:0.35 }}
    }});
  }}
  viewer.render();
}}

function resetView() {{
  if (!viewer) return;
  viewer.zoomTo({{model: window._ligIdx||0}});
  viewer.render();
}}

function toggleSurface() {{
  if (!viewer) return;
  if (!surfaceOn) {{
    viewer.addSurface($3Dmol.SurfaceType.VDW,
      {{ opacity:0.55, color:'#00d4aa' }},
      {{ model: window._ligIdx||0 }});
    surfaceOn = true;
    document.getElementById('surfBtn').classList.add('active');
  }} else {{
    viewer.removeAllSurfaces();
    surfaceOn = false;
    document.getElementById('surfBtn').classList.remove('active');
  }}
  viewer.render();
}}

function toggleSpin() {{
  if (!viewer) return;
  if (!spinning) {{
    viewer.spin(true);
    spinning = true;
    document.getElementById('spinBtn').classList.add('active');
  }} else {{
    viewer.spin(false);
    spinning = false;
    document.getElementById('spinBtn').classList.remove('active');
  }}
}}

function toggleLigStyle() {{
  if (!viewer) return;
  ligStyle = (ligStyle + 1) % 3;
  setLigandStyle(window._ligIdx||0, ligStyle);
  var labels = ['⬡ Ball+Stick','— Stick','● Sphere'];
  document.getElementById('styleBtn').textContent = labels[ligStyle];
}}

function setReceptorOpacity(val) {{
  if (!viewer) return;
  recOpacity = val;
  document.getElementById('opVal').textContent = Math.round(val*100) + '%';
  if (hasRec) {{
    viewer.setStyle({{model:0}}, {{
      cartoon: {{ color:'spectrum', opacity:val, thickness:0.4 }}
    }});
    viewer.render();
  }}
}}

if (typeof $3Dmol !== 'undefined' && $3Dmol.createViewer) {{
  initViewer();
}} else {{
  window.addEventListener('load', function() {{
    // Poll until 3Dmol is ready — CDN may load after window.load
    var attempts = 0;
    var poll = setInterval(function() {{
      attempts++;
      if (typeof $3Dmol !== 'undefined' && $3Dmol.createViewer) {{
        clearInterval(poll);
        initViewer();
      }}
      if (attempts > 50) {{
        clearInterval(poll);
        document.getElementById('status').textContent = 'Error: 3Dmol CDN failed to load';
        document.getElementById('status').style.background = '#450a0a';
        document.getElementById('status').style.color = '#f87171';
      }}
    }}, 100);
  }});
}}
</script>
</body>
</html>"""
    return html



# ═══════════════════════════════════════════════════════════════════════════════
# AUTO SETUP: grid calculation + initial docking from SMILES only
# ═══════════════════════════════════════════════════════════════════════════════

def auto_grid_from_receptor(receptor_pdbqt_path: str, padding: float = 8.0) -> dict:
    """
    Calculate a docking grid that covers the entire receptor with padding.
    Used when no binding site coordinates are known.
    Returns grid dict with center_x/y/z and size_x/y/z.
    """
    xs, ys, zs = [], [], []
    with open(receptor_pdbqt_path) as f:
        for line in f:
            if line.startswith(("ATOM","HETATM")):
                try:
                    xs.append(float(line[30:38]))
                    ys.append(float(line[38:46]))
                    zs.append(float(line[46:54]))
                except:
                    pass
    if not xs:
        raise ValueError("No atoms found in receptor PDBQT file.")
    cx = round((max(xs)+min(xs))/2, 3)
    cy = round((max(ys)+min(ys))/2, 3)
    cz = round((max(zs)+min(zs))/2, 3)
    sx = round(max(xs)-min(xs)+padding, 1)
    sy = round(max(ys)-min(ys)+padding, 1)
    sz = round(max(zs)-min(zs)+padding, 1)
    return {"center_x":cx,"center_y":cy,"center_z":cz,
            "size_x":sx,"size_y":sy,"size_z":sz}


def run_initial_dock(
    smiles: str,
    receptor_path: Path,
    grid: dict,
    work_dir: Path,
    exhaustiveness: int = 8,
    name: str = "initial",
) -> tuple[float | None, Path | None, Path | None]:
    """
    Convert SMILES → PDBQT and dock against receptor.
    Returns (best_affinity, docked_pdbqt_path, log_path).
    """
    pdbqt = smiles_to_pdbqt(smiles, name, work_dir)
    if pdbqt is None:
        return None, None, None
    affinity, docked = run_vina(pdbqt, name, work_dir, str(receptor_path), grid, exhaustiveness)
    log_path = work_dir / f"{name}_vina.log"
    return affinity, docked, log_path if log_path.exists() else None


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE CYCLE RUNNER (reusable in loop)
# ═══════════════════════════════════════════════════════════════════════════════

def run_one_cycle(
    smiles: str,
    receptor_path: Path,
    grid: dict,
    grid_centre: tuple,
    grid_size: tuple,
    target_name: str,
    pdb_id: str,
    cavity_desc: str,
    key_res_list: list,
    biology: str,
    chembl_context: str,
    resolved_chembl_id: str,
    notes_input: str,
    n_suggestions: int,
    backend: str,
    model_override: str,
    exhaustiveness: int,
    history: list,
    target_aff: float,
    mode: str,
    tmp_path: Path,
    ligand_pdbqt_bytes: bytes | None = None,
    numbering_offset: int = 0,
    cycle_num: int = 1,
    vina_seed: int = 0,
) -> dict:
    """
    Run one full pipeline cycle. Returns dict with results.
    """
    # Residue detection from provided PDBQT bytes or skip
    residues, contact_detail, ligand_centre = [], {}, None
    ligand_pdbqt_path = tmp_path / f"ligand_c{cycle_num}.pdbqt"
    if ligand_pdbqt_bytes:
        ligand_pdbqt_path.write_bytes(ligand_pdbqt_bytes)
        try:
            residues, contact_detail = detect_contacts(
                str(ligand_pdbqt_path), str(receptor_path),
                numbering_offset=numbering_offset
            )
            ligand_centre = get_ligand_centre(str(ligand_pdbqt_path))
        except Exception:
            pass

    # Resolve mode
    iter_mode, iter_msg = resolve_mode(mode, history, target_aff)

    # Build system prompt + user prompt
    system_prompt = build_system_prompt(
        target_name=target_name, pdb_id=pdb_id,
        cavity_description=cavity_desc or "Binding site defined by docking grid",
        key_residues=key_res_list, grid_centre=grid_centre, grid_size=grid_size,
        target_biology=biology, chembl_id=resolved_chembl_id,
    )
    full_notes = (notes_input + "\n" + chembl_context).strip() if chembl_context else notes_input
    prompt = build_prompt(
        smiles=smiles, docking_score=history[-1] if history else 0.0,
        residues=residues, contact_detail=contact_detail,
        ligand_centre=ligand_centre, grid_centre=grid_centre, grid_size=grid_size,
        compound_name=f"cycle_{cycle_num}", notes=full_notes,
        n_suggestions=n_suggestions, admet_mode=(iter_mode == "admet"),
    )

    # Call LLM
    suggestions_text = call_llm(prompt, system_prompt, backend, model_override or "")

    # Validate
    suggestions_annotated, validation_results = validate_response_smiles(suggestions_text)

    # Dock all valid
    docking_results = []
    best_affinity   = None
    best_smiles     = smiles
    best_pdbqt_bytes= ligand_pdbqt_bytes

    dock_dir = tmp_path / f"dock_c{cycle_num}"
    dock_dir.mkdir(exist_ok=True)

    for i, result in enumerate(validation_results, 1):
        if not result["valid"]:
            docking_results.append({**result, "docked": False, "reason": "invalid"})
            continue
        name_i   = f"c{cycle_num}_s{i}"
        pdbqt_out = smiles_to_pdbqt(result["smiles"], name_i, dock_dir)
        if pdbqt_out is None:
            docking_results.append({**result, "docked": False, "reason": "obabel failed"})
            continue

        # Add seed to grid for reproducibility
        grid_with_seed = {**grid, "seed": vina_seed}
        affinity, docked = run_vina(pdbqt_out, name_i, dock_dir, str(receptor_path), grid, exhaustiveness)
        if docked is None:
            docking_results.append({**result, "docked": False, "reason": "Vina failed"})
            continue

        docking_results.append({
            **result, "docked": True, "affinity": affinity,
            "pdbqt_path": str(docked), "compound_name": name_i,
        })

        if affinity and (best_affinity is None or abs(affinity) > abs(best_affinity)):
            best_affinity    = affinity
            best_smiles      = result["smiles"]
            best_pdbqt_bytes = docked.read_bytes()

    return {
        "cycle":             cycle_num,
        "input_smiles":      smiles,
        "suggestions_text":  suggestions_text,
        "validation_results":validation_results,
        "docking_results":   docking_results,
        "best_affinity":     best_affinity,
        "best_smiles":       best_smiles,
        "best_pdbqt_bytes":  best_pdbqt_bytes,
        "iter_mode":         iter_mode,
        "iter_msg":          iter_msg,
        "contact_detail":    contact_detail,
        "residues":          residues,
        "ligand_centre":     ligand_centre,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-title">AI-Guided Docking Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Structure-based drug discovery · Any protein target · AutoDock Vina · Claude AI · v2 with autonomous loop + 3D viewer</div>', unsafe_allow_html=True)

if not PIPELINE_LOADED:
    st.error(f"❌ Could not load docking_pipeline.py: {PIPELINE_ERROR}")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🎯 Target Setup")
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    target_name  = st.text_input("Target protein name", value="MRP1 (ABCC1)")
    pdb_id       = st.text_input("PDB ID", value="5UJA")
    cavity_desc  = st.text_area("Binding site description", height=70,
                                 value="Inward-open transmembrane substrate cavity, bipartite hydrophobic cavity ~2000 Å³ at the interface of TM helices 4-6, 10-11 (wing I) and TM helices 6-11, 12-17 (wing II)")
    key_residues = st.text_input("Key residues (comma-separated)",
                                  value="Thr1241,Trp1245,Phe594,Met1092,Tyr1032,Lys332,Gln450")
    biology      = st.text_area("Target biology (optional)", height=50,
                                 value="MRP1 is an ABC efflux transporter that pumps chemotherapy drugs out of cancer cells causing multidrug resistance. Inhibiting the inward-open conformation prevents the conformational transition required for drug efflux.")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Docking Grid")
    auto_grid = st.toggle("Auto-calculate grid from receptor", value=True,
                           help="Automatically calculates a grid covering the full receptor. Turn off to specify coordinates manually.")
    if not auto_grid:
        col1, col2 = st.columns(2)
        with col1:
            cx = st.number_input("Centre X", value=94.806, format="%.3f")
            cy = st.number_input("Centre Y", value=59.056, format="%.3f")
            cz = st.number_input("Centre Z", value=56.653, format="%.3f")
        with col2:
            sx = st.number_input("Size X (Å)", value=21.92, format="%.2f")
            sy = st.number_input("Size Y (Å)", value=21.297, format="%.3f")
            sz = st.number_input("Size Z (Å)", value=23.806, format="%.3f")
    else:
        cx=cy=cz=sx=sy=sz = 0.0
        st.info("Grid will be calculated automatically from receptor dimensions.")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("### 📁 Pipeline Inputs")
    receptor_file = st.file_uploader("Receptor PDBQT *(required)*", type=["pdbqt"], key="receptor")
    smiles_input  = st.text_area("Starting SMILES *(required)*", height=80,
                                  value="CC(C)(C)c1ccc(NC2CCCCC2C(=O)CCSC(SCCc2ccc(F)c(S(=O)(=O)O)c2C(=O)O)c2ccc3ccccc3c2/C=C/C2=CC=CC3=CC=CC4=C(C(=O)O)C=CC(=C23)C4)cc1")
    st.markdown("**Optional — skip if you want the pipeline to dock automatically:**")
    log_file      = st.file_uploader("Vina Log File (leave blank to auto-dock first)", type=["txt","log"], key="log")
    pdbqt_file    = st.file_uploader("Docked Ligand PDBQT (leave blank to auto-dock first)", type=["pdbqt"], key="lpdbqt")
    compound_name = st.text_input("Compound Name", value="MK571_final")
    numbering_offset = st.number_input("Residue numbering offset", value=905, step=1)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("### 🔬 ChEMBL")
    use_chembl    = st.toggle("Fetch ChEMBL data", value=True)
    chembl_id     = st.text_input("ChEMBL Target ID", value="CHEMBL4523")
    chembl_search = st.text_input("Or search by name", placeholder="e.g. EGFR")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("### 🎛️ Settings")
    mode           = st.selectbox("Optimisation Mode", ["affinity","admet","auto"], index=0)
    target_aff     = st.number_input("Target affinity (abs kcal/mol)", value=13.0, format="%.1f")
    n_suggestions  = st.slider("Suggestions per cycle", 1, 5, 3)
    exhaustiveness = st.select_slider("Vina Exhaustiveness", [8,16,32], value=8)
    fix_seed       = st.toggle("Fix random seed (reproducible)", value=False,
                                help="Uses seed 42 for Vina. Makes docking scores reproducible between runs.")
    backend        = st.selectbox("LLM Backend", ["claude","ollama"], index=0)
    model_override = st.text_input("Model override (optional)", placeholder="claude-sonnet-4-6")
    notes_input    = st.text_area("Notes for AI", height=60)
    history_input  = st.text_input("Affinity history", value="9.12,10.33,12.28")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("### 🔁 Autonomous Loop")
    auto_loop      = st.toggle("Enable autonomous iterative loop", value=False,
                                help="Automatically feeds the best compound back as input for N cycles.")
    n_cycles       = st.slider("Number of cycles", 1, 10, 3,
                                disabled=not auto_loop,
                                help="Each cycle takes the best compound from the previous one.")
    plateau_stop   = st.toggle("Stop early if plateau detected", value=True,
                                disabled=not auto_loop,
                                help="Stops if improvement < 0.3 kcal/mol for 2 consecutive cycles.")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    run_button = st.button("🚀 Run Pipeline", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# IDLE
# ═══════════════════════════════════════════════════════════════════════════════
if not run_button:
    st.markdown("### How it works")
    cols = st.columns(6)
    steps = [
        ("01","Parse Log","Reads any Vina log"),
        ("02","Contacts","Auto-detects residues"),
        ("03","ChEMBL","Fetches real SAR data"),
        ("04","AI","Generates modifications"),
        ("05","Validate","RDKit checks SMILES"),
        ("06","Loop","Re-docks best compound"),
    ]
    for col,(num,title,desc) in zip(cols,steps):
        with col:
            st.markdown(f"""<div style="background:#13161e;border:1px solid #1e2330;border-radius:10px;padding:0.8rem">
                <div style="color:#00d4aa;font-weight:800;font-family:'JetBrains Mono',monospace">{num}</div>
                <div style="font-weight:700;margin:0.2rem 0;font-size:0.85rem">{title}</div>
                <div style="color:#64748b;font-size:0.72rem">{desc}</div>
            </div>""", unsafe_allow_html=True)
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
errors = []
if not receptor_file:        errors.append("Upload a receptor PDBQT file")
if not smiles_input.strip(): errors.append("Enter a SMILES string")
if not target_name:          errors.append("Enter a target protein name")
for e in errors:
    st.error(f"❌ {e}")
if errors:
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════════════════
tmp_dir      = tempfile.mkdtemp()
tmp_path     = Path(tmp_dir)
receptor_path= tmp_path / "receptor.pdbqt"
receptor_path.write_bytes(receptor_file.read())
receptor_bytes = receptor_path.read_bytes()

try:
    # ── Auto grid ──────────────────────────────────────────────────────────────
    if auto_grid:
        setup_status = st.empty()
        setup_status.info("⚙️ Calculating grid from receptor dimensions...")
        grid_dict = auto_grid_from_receptor(str(receptor_path))
        cx = grid_dict["center_x"]; cy = grid_dict["center_y"]; cz = grid_dict["center_z"]
        sx = grid_dict["size_x"];   sy = grid_dict["size_y"];   sz = grid_dict["size_z"]
        setup_status.success(f"✓ Grid: centre ({cx}, {cy}, {cz}) · size ({sx} × {sy} × {sz} Å)")
    
    grid        = {"center_x":cx,"center_y":cy,"center_z":cz,"size_x":sx,"size_y":sy,"size_z":sz}
    grid_centre = (cx,cy,cz)
    grid_size   = (sx,sy,sz)
    vina_seed   = 42 if fix_seed else 0
    key_res_list= [r.strip() for r in key_residues.split(",") if r.strip()] if key_residues else []

    # ── Auto initial dock if no log/pdbqt provided ─────────────────────────────
    if not log_file or not pdbqt_file:
        dock_status = st.empty()
        dock_status.info(f"🔬 No initial docking provided — docking `{compound_name}` against receptor...")
        init_affinity, init_docked, init_log = run_initial_dock(
            smiles=smiles_input.strip(),
            receptor_path=receptor_path,
            grid=grid,
            work_dir=tmp_path,
            exhaustiveness=exhaustiveness,
            name="initial",
        )
        if init_docked is None:
            st.error("❌ Initial docking failed. Check your SMILES string and receptor file.")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            st.stop()
        dock_status.success(f"✓ Initial docking complete — best affinity: {init_affinity:.2f} kcal/mol")
        log_bytes           = init_log.read_bytes() if init_log else b""
        ligand_pdbqt_bytes  = init_docked.read_bytes()
        log_path            = init_log or tmp_path / "initial_vina.log"
        if not log_bytes and init_affinity:
            # Write a minimal log so parse_vina_log works
            log_path.write_text(
                f"Ligand: initial.pdbqt\n"
                f"mode |   affinity | dist from best mode\n"
                f"     | (kcal/mol) | rmsd l.b.| rmsd u.b.\n"
                f"-----+------------+----------+----------\n"
                f"   1 {init_affinity:>12.3f}          0          0\n"
            )
    else:
        log_path = tmp_path / "input.log"
        log_path.write_bytes(log_file.read())
        ligand_pdbqt_bytes = pdbqt_file.read() if pdbqt_file else None

    log_data = parse_vina_log(str(log_path))

except Exception as e:
    st.error(f"Setup failed: {e}")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    st.stop()

history     = ([abs(float(x)) for x in history_input.split(",") if x.strip()]
               if history_input.strip() else [])
history.append(abs(log_data["best_affinity"]))

# ── ChEMBL (fetch once, reuse across cycles) ──────────────────────────────────
chembl_context     = ""
resolved_chembl_id = chembl_id.strip()
if use_chembl:
    if not resolved_chembl_id and chembl_search.strip():
        found = search_chembl_target(chembl_search.strip())
        if found:
            resolved_chembl_id = found
    if resolved_chembl_id:
        compounds = fetch_chembl_inhibitors(resolved_chembl_id)
        if compounds:
            chembl_context = format_chembl_context(compounds)
            st.success(f"✓ ChEMBL: {len(compounds)} inhibitors for {resolved_chembl_id} "
                       f"(best pChEMBL={compounds[0]['pchembl']:.2f})")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════
n_total_cycles = n_cycles if auto_loop else 1
current_smiles = smiles_input.strip()
current_pdbqt  = ligand_pdbqt_bytes
all_cycle_results = []
overall_best_affinity = abs(log_data["best_affinity"])
overall_best_smiles   = current_smiles
overall_best_pdbqt    = ligand_pdbqt_bytes

progress = st.progress(0)
status   = st.empty()

try:
    for cycle_i in range(1, n_total_cycles + 1):

        if auto_loop:
            st.markdown(f"""<div class="cycle-header">
                <span style="color:#00d4aa;font-family:'JetBrains Mono',monospace;font-weight:800">
                CYCLE {cycle_i} / {n_total_cycles}</span>
                &nbsp;&nbsp;|&nbsp;&nbsp;
                <span style="color:#e2e8f0">Input: {current_smiles[:60]}...</span>
            </div>""", unsafe_allow_html=True)

        status.markdown(f"**Cycle {cycle_i}** · Running pipeline...")

        cycle_result = run_one_cycle(
            smiles            = current_smiles,
            receptor_path     = receptor_path,
            grid              = grid,
            grid_centre       = grid_centre,
            grid_size         = grid_size,
            target_name       = target_name,
            pdb_id            = pdb_id or "unknown",
            cavity_desc       = cavity_desc,
            key_res_list      = key_res_list,
            biology           = biology,
            chembl_context    = chembl_context,
            resolved_chembl_id= resolved_chembl_id,
            notes_input       = notes_input,
            n_suggestions     = n_suggestions,
            backend           = backend,
            model_override    = model_override,
            exhaustiveness    = exhaustiveness,
            history           = history.copy(),
            target_aff        = target_aff,
            mode              = mode,
            tmp_path          = tmp_path,
            ligand_pdbqt_bytes= current_pdbqt,
            numbering_offset  = int(numbering_offset),
            cycle_num         = cycle_i,
            vina_seed         = vina_seed,
        )
        all_cycle_results.append(cycle_result)

        # Update history
        if cycle_result["best_affinity"]:
            history.append(abs(cycle_result["best_affinity"]))

        # ── Display this cycle's results ──────────────────────────────────────
        with st.expander(
            f"Cycle {cycle_i} results — best affinity: "
            f"{cycle_result['best_affinity']:.2f} kcal/mol"
            if cycle_result["best_affinity"] else f"Cycle {cycle_i} results",
            expanded=(cycle_i == n_total_cycles)
        ):
            # Suggestions
            st.markdown("**AI Suggestions:**")
            st.markdown(cycle_result["suggestions_text"])

            # Validation cards
            for j, r in enumerate(cycle_result["validation_results"], 1):
                if not r["valid"]:
                    st.markdown(f"""<div class="result-block error">
                        <span class="badge badge-invalid">INVALID</span>
                        <strong> Compound {j}</strong><br>
                        <span style="color:#94a3b8">{r['smiles'][:80]}</span>
                    </div>""", unsafe_allow_html=True)
                else:
                    p = r["props"]
                    has_flag = any("FLAG" in x for x in r["issues"])
                    has_warn = any("WARN" in x for x in r["issues"])
                    badge_cls = "badge-flag" if has_flag else ("badge-warn" if has_warn else "badge-pass")
                    badge_txt = "FLAG" if has_flag else ("WARN" if has_warn else "PASS")
                    card_cls  = "error" if has_flag else ("warn" if has_warn else "")
                    affstr    = ""
                    for dr in cycle_result["docking_results"]:
                        if dr.get("smiles") == r["smiles"] and dr.get("affinity"):
                            affstr = f" · <strong>ΔG = {dr['affinity']:.2f} kcal/mol</strong>"
                    st.markdown(f"""<div class="result-block {card_cls}">
                        <span class="badge {badge_cls}">{badge_txt}</span>
                        <strong> Compound {j}</strong>{affstr}<br>
                        <span style="color:#94a3b8;font-size:0.75rem">{r['smiles'][:80]}</span><br>
                        MW={p.get('MW','?')} | cLogP={p.get('cLogP','?')} | TPSA={p.get('TPSA','?')} Å²
                    </div>""", unsafe_allow_html=True)

            # 3D Viewer for best docked pose this cycle
            if cycle_result["best_pdbqt_bytes"]:
                st.markdown("**3D Pose Viewer — Best compound this cycle:**")
                ligand_pdbqt_str = cycle_result["best_pdbqt_bytes"].decode("utf-8", errors="replace")
                receptor_str_full = receptor_bytes.decode("utf-8", "replace")
                viewer_html = render_3dmol(ligand_pdbqt_str, receptor_str_full)
                st.components.v1.html(viewer_html, height=440, scrolling=False)
                st.caption("Ligand shown as sticks · Receptor shown as cartoon · Scroll to zoom · Drag to rotate")

            # Download buttons
            if cycle_result["best_pdbqt_bytes"]:
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button(
                        f"⬇ Best pose (cycle {cycle_i})",
                        data=cycle_result["best_pdbqt_bytes"],
                        file_name=f"{compound_name}_cycle{cycle_i}_best.pdbqt",
                        mime="chemical/x-pdbqt",
                        key=f"dl_pose_{cycle_i}",
                    )
                with c2:
                    report_i = (
                        f"CYCLE {cycle_i} REPORT\n"
                        f"Input SMILES: {cycle_result['input_smiles']}\n"
                        f"Best affinity: {cycle_result['best_affinity']}\n"
                        f"Best SMILES: {cycle_result['best_smiles']}\n\n"
                        f"SUGGESTIONS:\n{cycle_result['suggestions_text']}\n"
                    )
                    st.download_button(
                        f"⬇ Report (cycle {cycle_i})",
                        data=report_i,
                        file_name=f"{compound_name}_cycle{cycle_i}_report.txt",
                        mime="text/plain",
                        key=f"dl_report_{cycle_i}",
                    )

        # ── Update for next cycle ─────────────────────────────────────────────
        if cycle_result["best_affinity"] and cycle_result["best_smiles"] != current_smiles:
            if abs(cycle_result["best_affinity"]) > overall_best_affinity:
                overall_best_affinity = abs(cycle_result["best_affinity"])
                overall_best_smiles   = cycle_result["best_smiles"]
                overall_best_pdbqt    = cycle_result["best_pdbqt_bytes"]
            current_smiles = cycle_result["best_smiles"]
            current_pdbqt  = cycle_result["best_pdbqt_bytes"]
        else:
            # No improvement — if plateau stop enabled, break
            if auto_loop and plateau_stop and cycle_i > 1:
                st.warning(f"⏹ Plateau detected at cycle {cycle_i} — stopping early. "
                           f"Best: {overall_best_affinity:.2f} kcal/mol")
                break

        progress.progress(int(cycle_i / n_total_cycles * 95))

    # ═══════════════════════════════════════════════════════════════════════════
    # OVERALL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("### 📊 Overall Summary")

    # Affinity progression table
    if len(all_cycle_results) > 1:
        rows = [f"| Cycle | Best Affinity | SMILES (truncated) |",
                f"|---|---|---|"]
        rows.append(f"| 0 (input) | {log_data['best_affinity']:.2f} | {smiles_input[:50]}... |")
        for cr in all_cycle_results:
            aff_str = f"{cr['best_affinity']:.2f}" if cr["best_affinity"] else "—"
            rows.append(f"| {cr['cycle']} | {aff_str} | {cr['best_smiles'][:50]}... |")
        st.markdown("\n".join(rows))

    # Final best card
    delta = overall_best_affinity - abs(log_data["best_affinity"])
    st.markdown(f"""<div class="stat-card" style="border-left:3px solid #00d4aa">
        <div class="stat-label">Overall Best Affinity</div>
        <div class="affinity-big">-{overall_best_affinity:.2f} kcal/mol</div>
        <div style="color:#34d399;font-family:'JetBrains Mono',monospace;font-size:0.85rem">
        +{delta:.2f} kcal/mol improvement over input</div>
        <div style="color:#64748b;font-size:0.78rem;margin-top:0.3rem">
        Best SMILES: {overall_best_smiles[:80]}...</div>
    </div>""", unsafe_allow_html=True)

    # Final 3D viewer
    if overall_best_pdbqt:
        st.markdown("### 🔬 Final Best Pose — 3D Viewer")
        ligand_str  = overall_best_pdbqt.decode("utf-8", errors="replace")
        receptor_str_full = receptor_bytes.decode("utf-8", "replace")
        viewer_html = render_3dmol(ligand_str, receptor_str_full, height=520)
        st.components.v1.html(viewer_html, height=540, scrolling=False)
        st.caption("Best overall compound · Receptor context shown · Use controls to toggle surface and spin")

        # Final downloads
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "⬇ Best Overall PDBQT",
                data=overall_best_pdbqt,
                file_name=f"{compound_name}_BEST_OVERALL.pdbqt",
                mime="chemical/x-pdbqt",
                use_container_width=True,
            )
        with dl2:
            full_report = (
                f"AI-GUIDED DOCKING — FULL LOOP REPORT\n"
                f"Target: {target_name} ({pdb_id})\n"
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                f"Cycles run: {len(all_cycle_results)}\n"
                f"Input affinity: {log_data['best_affinity']:.2f} kcal/mol\n"
                f"Best affinity: -{overall_best_affinity:.2f} kcal/mol\n"
                f"Improvement: +{delta:.2f} kcal/mol\n"
                f"Best SMILES: {overall_best_smiles}\n\n"
            )
            for cr in all_cycle_results:
                full_report += (
                    f"{'='*60}\nCYCLE {cr['cycle']}\n{'='*60}\n"
                    f"Input SMILES: {cr['input_smiles']}\n"
                    f"Best affinity: {cr['best_affinity']}\n\n"
                    f"{cr['suggestions_text']}\n\n"
                )
            st.download_button(
                "⬇ Full Loop Report (.txt)",
                data=full_report,
                file_name=f"{compound_name}_{target_name.replace(' ','_')}_loop_report.txt",
                mime="text/plain",
                use_container_width=True,
            )

    progress.progress(100)
    status.markdown("✅ Pipeline complete.")

except Exception as e:
    st.error(f"Pipeline error: {e}")
    import traceback
    st.code(traceback.format_exc())

finally:
    shutil.rmtree(tmp_dir, ignore_errors=True)
