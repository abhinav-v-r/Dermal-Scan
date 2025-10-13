#!/usr/bin/env python3
"""
Test script for Enhanced DermalScan AI
This script tests all the new components and imports
"""

import sys
import os

def test_imports():
    """Test all imports required for the enhanced application"""
    print("🧪 Testing Enhanced DermalScan AI Components...")
    
    try:
        # Test core imports
        import streamlit as st
        print("✅ Streamlit imported successfully")
        
        import cv2
        print("✅ OpenCV imported successfully")
        
        import numpy as np
        print("✅ NumPy imported successfully")
        
        from PIL import Image
        print("✅ PIL/Pillow imported successfully")
        
        import pandas as pd
        print("✅ Pandas imported successfully")
        
        # Test original modules
        from image_load import loader, preprocess, label
        print("✅ Image processing modules imported successfully")
        
        from models import predict_age, predict_feature
        print("✅ AI model modules imported successfully")
        
        # Test new components
        from components import (
            load_css, render_header, render_navigation, render_features_grid,
            render_sample_gallery, render_dermatologist_notes, render_about_section,
            render_results_section, render_recommendations, render_footer
        )
        print("✅ Professional UI components imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_file_structure():
    """Test that all required files exist"""
    print("\n📁 Testing File Structure...")
    
    required_files = [
        "app.py",
        "components.py",
        "requirements.txt",
        "assets/css/styles.css",
        "image_load/loader.py",
        "image_load/preprocess.py",
        "image_load/label.py",
        "models/predict_age.py",
        "models/predict_feature.py"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def test_sample_images():
    """Test sample images directory"""
    print("\n🖼️ Testing Sample Images...")
    
    sample_dir = "sample_images"
    if os.path.exists(sample_dir):
        images = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"✅ Sample images directory exists with {len(images)} images")
        for img in images[:5]:  # Show first 5
            print(f"  📸 {img}")
        if len(images) > 5:
            print(f"  ... and {len(images) - 5} more")
        return True
    else:
        print("❌ Sample images directory not found")
        return False

def test_css_loading():
    """Test CSS file accessibility"""
    print("\n🎨 Testing CSS Styling...")
    
    css_path = "assets/css/styles.css"
    if os.path.exists(css_path):
        with open(css_path, 'r') as f:
            content = f.read()
            if len(content) > 1000:  # Should be a substantial CSS file
                print(f"✅ CSS file loaded ({len(content)} characters)")
                return True
            else:
                print(f"⚠️ CSS file seems small ({len(content)} characters)")
                return False
    else:
        print("❌ CSS file not found")
        return False

def main():
    """Run all tests"""
    print("🧑‍⚕️ DermalScan AI - Enhanced Edition Test Suite")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("File Structure Test", test_file_structure),
        ("Sample Images Test", test_sample_images),
        ("CSS Loading Test", test_css_loading)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your enhanced DermalScan AI is ready to run.")
        print("\n🚀 To start the application, run:")
        print("   py -m streamlit run app.py")
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        if passed >= total - 1:
            print("🔧 Minor issues detected, but the app should still work.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)