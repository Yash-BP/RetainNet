from setuptools import setup, find_packages

setup(
    name="retainnet",
    version="1.0.0",
    description="RetainNet: Deep Survival Analysis Framework for Dynamic Customer Lifetime Value (CLTV) & Proactive Churn Mitigation",
    packages=find_packages(),
    py_modules=["dashboard", "main"],
    include_package_data=True,
    install_requires=[
        "numpy>=1.23.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.2.0",
        "scipy>=1.9.0",
        "torch>=2.0.0",
        "lifelines>=0.27.0",
        "streamlit>=1.28.0",
    ],
    python_requires=">=3.10",
)
