"""ParaView-compatible VTU vertices with appended raw little-endian arrays."""
from pathlib import Path
import struct
from xml.sax.saxutils import escape
import numpy as np


def write_vtu(path, points, point_data):
    path=Path(path)
    n=len(points)
    arrays=[]
    offset=0
    def desc(name, values, vtk_type, components=1):
        nonlocal offset
        a=np.ascontiguousarray(values)
        arrays.append(a)
        text=f'<DataArray type="{vtk_type}" Name="{escape(name)}" NumberOfComponents="{components}" format="appended" offset="{offset}"/>'
        offset+=8+a.nbytes
        return text
    fields=[]
    for name,values in point_data.items():
        a=np.asarray(values)
        if len(a)!=n: raise ValueError(f'Wrong point field length: {name}')
        integer=np.issubdtype(a.dtype,np.integer)
        a=a.astype('<i8' if integer else '<f4',copy=False)
        fields.append(desc(name,a,'Int64' if integer else 'Float32',a.shape[1] if a.ndim==2 else 1))
    pdesc=desc('Points',np.asarray(points,dtype='<f4'),'Float32',3)
    cells=[desc('connectivity',np.arange(n,dtype='<i8'),'Int64'),
           desc('offsets',np.arange(1,n+1,dtype='<i8'),'Int64'),
           desc('types',np.ones(n,dtype='u1'),'UInt8')]
    header=('<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">'
            f'<UnstructuredGrid><Piece NumberOfPoints="{n}" NumberOfCells="{n}">'
            '<PointData>'+''.join(fields)+'</PointData><CellData/>'
            '<Points>'+pdesc+'</Points><Cells>'+''.join(cells)+'</Cells></Piece></UnstructuredGrid>'
            '<AppendedData encoding="raw">_').encode('utf-8')
    tmp=path.with_suffix('.vtu.tmp')
    with tmp.open('wb') as f:
        f.write(header)
        for a in arrays:
            f.write(struct.pack('<Q',a.nbytes))
            f.write(memoryview(a).cast('B'))
        f.write(b'</AppendedData></VTKFile>')
    tmp.replace(path)


def write_pvd(path, entries):
    from xml.etree.ElementTree import Element,SubElement,ElementTree
    root=Element('VTKFile',type='Collection',version='0.1',byte_order='LittleEndian')
    collection=SubElement(root,'Collection')
    for progress,filename in entries:
        SubElement(collection,'DataSet',timestep=str(progress),group='',part='0',file=filename)
    temp=Path(path).with_suffix('.pvd.tmp')
    ElementTree(root).write(temp,encoding='utf-8',xml_declaration=True)
    temp.replace(path)
