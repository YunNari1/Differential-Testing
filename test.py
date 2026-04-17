import os
import sys

def run_command(cmd):
    print("=" * 50)
    print(f"[RUN] {cmd}")
    print("=" * 50)

    result = os.system(cmd)

    if result != 0:
        print(f"[ERROR] Command failed: {cmd}")
        sys.exit(1)


def check_models():
    if not os.path.exists("model1.h5") or not os.path.exists("model2.h5"):
        print("[INFO] Model files not found. Training models first...")
        run_command("python deepxplore/CIFAR10/train_models.py")
    else:
        print("[INFO] Found existing models. Skipping training.")
 


def run_deepxplore():
    print("[INFO] Running DeepXplore differential testing...")

    cmd = "python deepxplore/CIFAR10/gen_diff.py light 1 1 10 50 10 0.2"
    run_command(cmd)


def check_results():
    folder = "results"

    if not os.path.exists(folder):
        print("[WARNING] No results folder found.")
        return

    files = [f for f in os.listdir(folder) if f.endswith(".png")]

    print("=" * 50)
    print(f"[RESULT] Generated {len(files)} disagreement cases.")
    print("=" * 50)

    if len(files) == 0:
        print("[WARNING] No images generated. Check your setup.")
    else:
        print("[INFO] Sample outputs:")
        for f in files[:5]:
            print(" -", f)


def main():
    print("\nDeepXplore CIFAR-10 Test Script\n")

    # 1. 모델 준비
    check_models()

    # 2. DeepXplore 실행
    run_deepxplore()

    # 3. 결과 확인
    check_results()

    print("\n[INFO] Done. Check 'results/' folder.\n")


if __name__ == "__main__":
    main()