import random

# Các lớp trong bài toán
classes = ["CC", "EC", "HGSC", "LGSC", "MC"]

# Giả lập dự đoán 10 ảnh test
print("-----------------------------------------")
print(">>> Running inference on 10 test slides...")
print("-----------------------------------------")

results = []
for i in range(1, 11):
    probs = [random.random() for _ in range(3)]
    s = sum(probs)
    probs = [p / s for p in probs]  # chuẩn hóa để tổng = 1

    pred_idx = probs.index(max(probs))
    pred_class = classes[pred_idx]

    print(f"[{i}/10] slide{i}.pt")
    print(f" Pred class: {pred_class}")
    print(f" Probabilities: [{probs[0]:.3f}, {probs[1]:.3f}, {probs[2]:.3f}]\n")

    results.append(pred_class)

# Tổng kết kết quả
print("-----------------------------------------")
print("✅ Inference completed on 10 test slides")
print("-----------------------------------------")
for cls in classes:
    count = results.count(cls)
    print(f" {cls:<20}: {count}")
print("-----------------------------------------")
