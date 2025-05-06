import os
import streamlit as st
from git import Repo

# Usar token de GitHub desde secrets
token = st.secrets["GITHUB_TOKEN"]
username = st.secrets["GITHUB_USERNAME"]

# Ruta del repo privado
repo_url = f"https://{username}:{token}@github.com/{username}/neu_mm.git"

# Ruta local para clonar
clone_dir = "/tmp/neu_mm"

# Clonar si no existe
if not os.path.exists(clone_dir):
    Repo.clone_from(repo_url, clone_dir)

# Ahora puedes acceder a tu modelo:
model_path = os.path.join(clone_dir, "modelo_neumonia.keras")
