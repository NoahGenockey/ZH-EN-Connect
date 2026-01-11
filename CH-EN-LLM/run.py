"""
Run Script - Quick Launcher for LinguaBridge Components
Usage: python run.py [component]

Components:
  process    - Run data processing pipeline
  train      - Train teacher model (requires GPU)
  distill    - Distill student model (local)
  gui        - Launch GUI application
  api        - Launch REST API server
  test       - Test inference
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='LinguaBridge Local - Quick Launcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py process          # Process raw data
  python run.py distill          # Train student model locally
  python run.py gui              # Launch desktop application
  python run.py api              # Start API server
  python run.py test             # Test translation
        """
    )
    
    parser.add_argument(
        'component',
        choices=['process', 'train', 'distill', 'gui', 'api', 'test'],
        help='Component to run'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print(f"LinguaBridge Local - Running: {args.component}")
    print("="*60)
    print()
    
    if args.component == 'process':
        from src.data_processor import main as run_processor
        run_processor()
        
    elif args.component == 'train':
        print("⚠️  WARNING: This requires GPU server!")
        print("Ensure you're running on a cloud instance with CUDA GPU.")
        response = input("Continue? (y/N): ")
        if response.lower() == 'y':
            from src.train_teacher import main as run_teacher
            run_teacher()
        else:
            print("Cancelled.")
            
    elif args.component == 'distill':
        print("Starting student model distillation...")
        print("This will run on CPU and may take several hours.")
        from src.distill_local import main as run_distill
        run_distill()
        
    elif args.component == 'gui':
        from src.app_gui import main as run_gui
        run_gui()
        
    elif args.component == 'api':
        from src.app_api import main as run_api
        run_api()
        
    elif args.component == 'test':
        print("Testing inference engine...\n")
        from src.inference import TranslationInference
        from src.utils import load_config
        
        config = load_config()
        
        try:
            print("Loading model...")
            engine = TranslationInference(config)
            
            test_sentences = [
                "Hello, world!",
                "How are you today?",
                "This is a test of the translation system.",
                "The weather is nice today."
            ]
            
            print("\nRunning test translations:\n")
            for text in test_sentences:
                print(f"EN: {text}")
                translation = engine.translate(text)
                print(f"ZH: {translation}")
                print()
            
            print("✅ Inference test complete!")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print("\nMake sure you have:")
            print("1. Processed data in data/processed/")
            print("2. Trained model in models/student/final_model/")
            sys.exit(1)


if __name__ == "__main__":
    main()
