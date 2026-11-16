"""Export vector figure text as outlines, as required by the AAAI author kit."""
from pathlib import Path
import shutil
import subprocess
import tempfile


def save_outlined_pdf(fig, path, **savefig_kwargs):
    gs = shutil.which('gs')
    if gs is None:
        raise RuntimeError('Ghostscript is required to export figure text as vector outlines.')
    with tempfile.TemporaryDirectory(prefix='aaai-figure-') as td:
        source = Path(td) / 'text.pdf'
        fig.savefig(source, **savefig_kwargs)
        subprocess.run([gs, '-q', '-dBATCH', '-dNOPAUSE', '-sDEVICE=pdfwrite',
                        '-dNoOutputFonts', '-dCompatibilityLevel=1.7',
                        f'-sOutputFile={path}', str(source)], check=True)
