"""
Quick test to verify the pipeline setup is correct.
"""
import sys
from pathlib import Path

def test_imports():
    """Test that all required packages are available."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not found")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError:
        print("✗ Transformers not found")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError:
        print("✗ NumPy not found")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError:
        print("✗ Matplotlib not found")
        return False
    
    return True

def test_data_structure():
    """Test that data directories and templates exist."""
    print("\nTesting data structure...")
    
    required_dirs = [
        "data/entities",
        "data/templates",
        "data/qa_templates"
    ]
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ not found")
            return False
    
    # Check for template files
    template_files = [
        "data/qa_templates/birth_date_questions.txt",
        "data/qa_templates/birth_city_questions.txt",
        "data/qa_templates/university_questions.txt",
        "data/qa_templates/major_questions.txt",
        "data/qa_templates/employer_questions.txt",
        "data/qa_templates/company_city_questions.txt",
    ]
    
    for file_path in template_files:
        if Path(file_path).exists():
            # Check it has at least 3 templates
            with open(file_path) as f:
                lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
                if len(lines) >= 3:
                    print(f"✓ {file_path} ({len(lines)} templates)")
                else:
                    print(f"⚠ {file_path} has only {len(lines)} templates (need ≥3)")
        else:
            print(f"✗ {file_path} not found")
            return False
    
    return True

def test_scripts():
    """Test that all pipeline scripts exist."""
    print("\nTesting scripts...")
    
    scripts = [
        "scripts/01_generate_dataset.py",
        "scripts/02_sft_base_model.py",
        "scripts/03_collect_activations.py",
        "scripts/04_train_sae.py",
        "scripts/05_evaluate_sae.py",
    ]
    
    for script in scripts:
        if Path(script).exists():
            print(f"✓ {script}")
        else:
            print(f"✗ {script} not found")
            return False
    
    return True

def main():
    print("="*60)
    print("1-to-1 SAE Pipeline Setup Test")
    print("="*60)
    
    all_pass = True
    
    all_pass &= test_imports()
    all_pass &= test_data_structure()
    all_pass &= test_scripts()
    
    print("\n" + "="*60)
    if all_pass:
        print("✅ All tests passed! Ready to run pipeline.")
        print("\nNext steps:")
        print("  python scripts/01_generate_dataset.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Check that data/qa_templates/ has all 6 question files")
    print("="*60)

if __name__ == "__main__":
    main()
