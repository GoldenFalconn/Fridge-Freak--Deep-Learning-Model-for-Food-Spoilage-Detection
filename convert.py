import jupytext

nb = jupytext.read("project.py")  # Read .py file
jupytext.write(nb, "project_converted.ipynb")  # Save as .ipynb
