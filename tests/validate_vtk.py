"""Independent VTK reader validation (author QA environment has PyVista)."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pyvista as pv

ROOT=Path(__file__).resolve().parents[1]
report=json.loads((ROOT/'qa/integration_report.json').read_text())
results=[]
for run in report['tests']:
    for path in (ROOT/run['output']).rglob('*.vtu'):
        mesh=pv.read(path)
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:
            checks={
                'point_count':mesh.n_points==len(z['points_reference_mm']),
                'reference_geometry':np.array_equal(mesh.points,z['points_reference_mm']),
                'particle_rows':np.array_equal(mesh['particle_row_id'],z['particle_row_id']),
                'predicted_displacement':np.array_equal(mesh['displacement_prediction_mm'],z['displacement_prediction_mm']),
                'predicted_strain':np.array_equal(mesh['equivalent_strain_prediction'],z['equivalent_strain_prediction']),
                'predicted_stress':np.array_equal(mesh['equivalent_stress_prediction_MPa'],z['equivalent_stress_prediction_MPa']),
                'reference_stress':np.array_equal(mesh['equivalent_stress_reference_MPa'],z['equivalent_stress_reference_MPa']),
            }
        assert all(checks.values()),(path,checks)
        results.append(dict(file=path.relative_to(ROOT).as_posix(),particles=mesh.n_points,checks=checks))
    for path in (ROOT/run['output']).rglob('*.pvd'):
        entries=ET.parse(path).findall('.//DataSet')
        times=[float(e.attrib['timestep']) for e in entries]
        assert times==sorted(times)
        assert all((path.parent/e.attrib['file']).is_file() for e in entries)
(ROOT/'qa/vtk_validation.json').write_text(json.dumps(dict(ok=True,reader='PyVista/VTK',files=results),indent=2),encoding='utf-8')
print(f'Independent VTK reader validated {len(results)} files and PVD sequences.')
