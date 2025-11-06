#!/usr/bin/env python3
"""
Training script for all embedding models with proper VRAM management.
Supports sequential training with memory cleanup between models.
"""

import os
import sys
import yaml
import time
import logging
import subprocess
import gc
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

class ModelTrainer:
    """Main class for training all embedding models sequentially"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.logger = self.setup_logging()
        self.results = {}
        
        # Model configurations
        self.model_configs = {
            'gemma': {
                'directory': 'google_gemma_embeddings',
                'supports_args': False,  # Gemma doesn't support command line args
                'description': 'Google Gemma Embeddings'
            },
            'bge_m3': {
                'directory': 'baai_bge_m3', 
                'supports_args': True,
                'description': 'BGE-M3 Multilingual Embeddings'
            },
            'minilm': {
                'directory': 'all_minilm_l6_v2',
                'supports_args': True,
                'description': 'MiniLM-L6-v2 Embeddings'
            }
        }
    
    def setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"train_all_models_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger = logging.getLogger(__name__)
        logger.info(f"Logging initialized. Log file: {log_file}")
        return logger
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"Error: Configuration file '{self.config_path}' not found!")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            sys.exit(1)
    
    def clear_vram(self):
        """Clear VRAM and system memory aggressively"""
        self.logger.info("Clearing VRAM and system memory...")
        
        try:
            # Try to import torch and clear CUDA cache aggressively
            import torch
            if torch.cuda.is_available():
                # Multiple rounds of cache clearing
                for i in range(3):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    time.sleep(1)
                
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()
                
                # Get memory info
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                
                self.logger.info(f"CUDA cache cleared - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
        except ImportError:
            self.logger.warning("PyTorch not available, skipping CUDA cache clear")
        
        # Aggressive garbage collection
        for i in range(3):
            gc.collect()
            time.sleep(1)
        
        # Additional cleanup for transformers/sentence-transformers
        try:
            import transformers
            # Clear transformers cache if available
            if hasattr(transformers, 'utils') and hasattr(transformers.utils, 'hub'):
                transformers.utils.hub.cached_file.cache_clear()
        except:
            pass
        
        # Wait longer for cleanup
        time.sleep(5)
        self.logger.info("Aggressive memory cleanup completed")
    
    def kill_old_training_processes(self):
        """Kill any old training processes that might be hanging"""
        try:
            import psutil
            current_pid = os.getpid()
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    # Skip current process
                    if proc.info['pid'] == current_pid:
                        continue
                    
                    # Look for Python processes running train.py
                    if (proc.info['name'] == 'python' and 
                        proc.info['cmdline'] and 
                        any('train.py' in arg for arg in proc.info['cmdline'])):
                        
                        self.logger.info(f"Killing old training process: PID {proc.info['pid']}")
                        proc.kill()
                        proc.wait(timeout=5)
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                    pass
                    
        except ImportError:
            self.logger.warning("psutil not available, skipping process cleanup")
    
    def get_enabled_models(self) -> List[str]:
        """Get list of enabled models from config"""
        enabled_models = []
        
        for model_name in self.model_configs.keys():
            model_config = self.config.get('models', {}).get(model_name, {})
            if model_config.get('enabled', False):
                enabled_models.append(model_name)
        
        return enabled_models
    
    def validate_model_directory(self, model_name: str) -> bool:
        """Validate that model directory and train.py exist"""
        model_info = self.model_configs[model_name]
        model_dir = Path(model_info['directory'])
        train_script = model_dir / 'train.py'
        
        if not model_dir.exists():
            self.logger.error(f"Model directory not found: {model_dir}")
            return False
        
        if not train_script.exists():
            self.logger.error(f"Training script not found: {train_script}")
            return False
        
        return True
    
    def build_command(self, model_name: str) -> List[str]:
        """Build training command for specific model"""
        model_info = self.model_configs[model_name]
        model_config = self.config['models'][model_name]
        
        cmd = ['python', 'train.py']
        
        # Only add arguments for models that support them
        if model_info['supports_args']:
            cmd.extend([
                '--batch_size', str(model_config.get('batch_size', 8)),
                '--learning_rate', str(model_config.get('learning_rate', 2e-5)),
                '--num_epochs', str(model_config.get('num_epochs', 3))
            ])
            
            # Add wandb flag if enabled
            if model_config.get('use_wandb', False):
                cmd.append('--use_wandb')
        
        return cmd
    
    def train_single_model(self, model_name: str) -> bool:
        """Train a single model"""
        model_info = self.model_configs[model_name]
        model_config = self.config['models'][model_name]
        
        self.logger.info("=" * 80)
        self.logger.info(f"Starting training: {model_info['description']}")
        self.logger.info(f"Model: {model_name}")
        self.logger.info(f"Directory: {model_info['directory']}")
        self.logger.info(f"Batch size: {model_config.get('batch_size', 'default')}")
        self.logger.info(f"Learning rate: {model_config.get('learning_rate', 'default')}")
        self.logger.info(f"Epochs: {model_config.get('num_epochs', 'default')}")
        self.logger.info("=" * 80)
        
        # Validate model directory
        if not self.validate_model_directory(model_name):
            return False
        
        # Build command
        cmd = self.build_command(model_name)
        model_dir = Path(model_info['directory'])
        
        self.logger.info(f"Executing command: {' '.join(cmd)}")
        self.logger.info(f"Working directory: {model_dir.absolute()}")
        
        try:
            # Start training
            start_time = time.time()
            
            result = subprocess.run(
                cmd,
                cwd=model_dir,
                capture_output=True,
                text=True
                # No timeout - let models train as long as needed
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Log results
            if result.returncode == 0:
                self.logger.info(f"{model_name.upper()} training completed successfully!")
                self.logger.info(f"Training duration: {duration:.2f} seconds")
                
                # Log some output for verification
                if result.stdout:
                    self.logger.info("Training output (last 500 chars):")
                    self.logger.info(result.stdout[-500:])
                
                self.results[model_name] = {
                    'status': 'success',
                    'duration': duration,
                    'message': 'Training completed successfully'
                }
                return True
            else:
                self.logger.error(f"{model_name.upper()} training failed!")
                self.logger.error(f"Return code: {result.returncode}")
                self.logger.error(f"Error output: {result.stderr}")
                
                self.results[model_name] = {
                    'status': 'failed',
                    'duration': duration,
                    'message': f"Training failed with return code {result.returncode}",
                    'error': result.stderr
                }
                return False
                
        except Exception as e:
            self.logger.error(f"{model_name.upper()} training failed with exception: {str(e)}")
            self.results[model_name] = {
                'status': 'error',
                'message': f"Exception occurred: {str(e)}"
            }
            return False
    
    def print_summary(self):
        """Print training summary"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TRAINING SUMMARY")
        self.logger.info("=" * 80)
        
        successful = 0
        failed = 0
        
        for model_name, result in self.results.items():
            status = result['status']
            duration = result.get('duration', 0)
            
            if status == 'success':
                self.logger.info(f"{model_name.upper()}: SUCCESS ({duration:.2f}s)")
                successful += 1
            else:
                self.logger.info(f"{model_name.upper()}: {status.upper()} - {result['message']}")
                failed += 1
        
        self.logger.info("-" * 80)
        self.logger.info(f"Total models: {len(self.results)}")
        self.logger.info(f"Successful: {successful}")
        self.logger.info(f"Failed: {failed}")
        
        if failed == 0:
            self.logger.info("🎉 All models trained successfully!")
        else:
            self.logger.warning(f"{failed} model(s) failed to train")
        
        self.logger.info("=" * 80)
    
    def run(self) -> int:
        """Main training loop"""
        self.logger.info("Starting sequential model training...")
        
        # Initial cleanup - kill old processes and clear memory
        self.logger.info("Performing initial cleanup...")
        self.kill_old_training_processes()
        self.clear_vram()
        
        # Get enabled models
        enabled_models = self.get_enabled_models()
        
        if not enabled_models:
            self.logger.error("No models are enabled in configuration!")
            return 1
        
        self.logger.info(f"Enabled models: {', '.join(enabled_models)}")
        
        # Train each model sequentially
        for i, model_name in enumerate(enabled_models, 1):
            self.logger.info(f"\n Training model {i}/{len(enabled_models)}: {model_name}")
            
            # ALWAYS clear VRAM before EVERY model (including first)
            self.logger.info("Pre-training cleanup...")
            self.clear_vram()
            self.kill_old_training_processes()
            time.sleep(2)
            
            # Train the model
            success = self.train_single_model(model_name)
            
            # Post-training cleanup after EVERY model
            self.logger.info("Post-training cleanup...")
            self.clear_vram()
            self.kill_old_training_processes()
            
            # Log completion message
            if i < len(enabled_models):
                self.logger.info(f"Completed {model_name}, preparing for next model...\n")
            else:
                self.logger.info(f"Completed {model_name} (final model)\n")
        
        # Print final summary
        self.print_summary()
        
        # Return appropriate exit code
        failed_count = sum(1 for result in self.results.values() if result['status'] != 'success')
        return 0 if failed_count == 0 else 1


def main():
    """Main entry point"""
    try:
        trainer = ModelTrainer()
        return trainer.run()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 130
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())