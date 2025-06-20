import re
from pathlib import Path

version = Path("VERSION.txt").read_text(encoding="utf-8").strip()
readme_path = Path("README.md")
readme_text = readme_path.read_text(encoding="utf-8")

# Substitui marcador {{VERSION}} ou qualquer versão anterior
new_readme = re.sub(
    r'(\*\*Version )(\{\{VERSION\}\}|\d+\.\d+\.\d+(?:\.\d+)?)(\*\*:)',
    rf'\1{version}\3',
    readme_text
)

readme_path.write_text(new_readme, encoding="utf-8")
print(f"README.md atualizado para a versão {version}")
