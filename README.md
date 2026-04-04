# S-VLB (Selective Vision-Language Backbone)

โปรเจกต์นี้เป็นโครงสร้างตั้งต้นสำหรับทำงานสาย Vision-Language แบบ hybrid โดยเอา CNN + Mamba + Transformer มาต่อกันในสไตล์ research code ที่แยกส่วนชัด ๆ แก้ต่อก็ง่าย เปลี่ยน backbone ก็ง่ายเหมือนกัน

ภาพรวมของ flow หลักคือ

1. ใช้ MobileNetV3 ดึง spatial feature จากภาพ
2. แปลง tensor จากรูปแบบ 4D `[B, C, H, W]` ไปเป็น sequence 3D `[B, N, C]`
3. ส่ง sequence เข้า Mamba เพื่อทำ selective sequence modeling
4. ใช้ Transformer encode ข้อความ
5. รวม vision tokens กับ text tokens ผ่าน fusion transformer
6. ส่งเข้า head เพื่อให้ score สำหรับงาน matching หรือ zero-shot inference

## โครงสร้างโปรเจกต์

```text
VLM-mamba/
├── configs/
│   └── default.yaml
├── data/
│   ├── __init__.py
│   └── image_text_dataset.py
├── models/
│   ├── __init__.py
│   ├── interfaces.py
│   ├── svlb.py
│   ├── backbones/
│   │   ├── __init__.py
│   │   ├── cnn.py
│   │   ├── mamba.py
│   │   └── transformer.py
│   ├── fusion/
│   │   ├── __init__.py
│   │   └── multimodal.py
│   └── heads/
│       ├── __init__.py
│       └── matching.py
├── scripts/
│   ├── run_inference.sh
│   └── smoke_test.sh
├── utils/
│   ├── __init__.py
│   ├── config.py
│   ├── metrics.py
│   ├── preprocessing.py
│   └── tensor_ops.py
├── .gitignore
├── inference.py
├── main.py
└── requirements.txt
```

## โฟลเดอร์นี้มีอะไรบ้าง

### `models/`

เก็บสถาปัตยกรรมทั้งหมดของโมเดล แยกเป็นชิ้น ๆ เพื่อให้ถอดเปลี่ยนได้ง่าย

- `models/__init__.py` export ตัวหลักของโมเดลไว้ใช้จากภายนอก
- `models/interfaces.py` รวม abstract interfaces สำหรับ vision backbone, sequence backbone, text backbone, fusion และ prediction head
- `models/svlb.py` ตัวประกอบโมเดลหลัก เอาทุกโมดูลมาต่อกันเป็น S-VLB

### `models/backbones/`

ส่วน backbone แต่ละชนิด

- `models/backbones/cnn.py` ใช้ MobileNetV3 เป็น image encoder แล้ว project channel ให้ตรงกับ embedding dim
- `models/backbones/mamba.py` เป็น sequence backbone ฝั่ง Mamba ถ้าเครื่องยังไม่มี `mamba_ssm` จะ fallback ไปใช้ selective SSM stub ที่รันได้ก่อน
- `models/backbones/transformer.py` เป็น text encoder แบบ Transformer สำหรับฝั่งข้อความ

### `models/fusion/`

ส่วนรวมข้อมูลหลาย modality

- `models/fusion/multimodal.py` รวม vision tokens กับ text tokens ด้วย Transformer encoder แล้วดึง embedding กลางออกมาใช้ต่อ

### `models/heads/`

ส่วนหัวสำหรับ prediction

- `models/heads/matching.py` head สำหรับให้ score งาน image-text matching หรือ zero-shot ranking

### `data/`

ส่วนโหลดข้อมูล

- `data/__init__.py` export dataset หลัก
- `data/image_text_dataset.py` dataset สำหรับ image-text pair โดยอ่าน annotation แบบ JSON Lines แล้วโหลดรูป + tokenize ข้อความให้พร้อมใช้งาน

ตัวอย่าง format ของ annotation แต่ละบรรทัด

```json
{ "image": "sample.jpg", "text": "a red car on the road" }
```

### `utils/`

ของเสริมที่ไม่ใช่ model โดยตรง

- `utils/config.py` โหลด config จาก YAML
- `utils/metrics.py` metric เบื้องต้น เช่น sigmoid confidence และ binary accuracy
- `utils/preprocessing.py` tokenizer แบบง่าย + image transform
- `utils/tensor_ops.py` จุดสำคัญสำหรับ reshape tensor ระหว่าง CNN กับ Mamba

### `configs/`

เก็บค่าคอนฟิกสำหรับทดลอง

- `configs/default.yaml` ค่า default ของ model, data และ inference

### `scripts/`

สคริปต์ช่วยรันจาก shell

- `scripts/smoke_test.sh` เรียก `main.py` เพื่อเช็กว่าโมเดลประกอบและรันได้
- `scripts/run_inference.sh` เรียก `inference.py` พร้อม image และ text จาก command line

## ไฟล์หลักตรง root

- `main.py` ใช้สำหรับ smoke test ด้วยข้อมูลสุ่ม เช็กว่า graph ของโมเดลต่อกันครบ
- `inference.py` ใช้รัน zero-shot inference กับรูปจริงและ candidate texts หลายประโยค
- `requirements.txt` dependency หลักของโปรเจกต์
- `.gitignore` กันไฟล์ cache, checkpoint, outputs และ artifact ที่ไม่ควรขึ้น git

## วิธีติดตั้ง

ถ้ายังไม่มี virtual environment แนะนำให้สร้างก่อน

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

ถ้าใช้ macOS แล้วติดตั้ง `mamba-ssm` ไม่ผ่าน ก็ยังรัน scaffold ได้อยู่ เพราะในโค้ดมี fallback ให้สำหรับการทดสอบ flow เบื้องต้น

## วิธีพัฒนา

แนวคิดของโปรเจกต์นี้คือ plug-and-play เพราะงั้นเวลาจะพัฒนาต่อ แนะนำ flow ประมาณนี้

1. เริ่มจากแก้ค่าที่ `configs/default.yaml` ก่อน เช่น embedding dim, depth, image size
2. ถ้าจะเปลี่ยน vision encoder ให้แก้หรือเพิ่มไฟล์ใน `models/backbones/` แล้วประกอบใหม่ใน `models/svlb.py`
3. ถ้าจะเปลี่ยน logic รวมภาพกับข้อความ ให้เพิ่ม fusion module ใน `models/fusion/`
4. ถ้าจะเปลี่ยน task จาก matching ไปเป็น classification หรือ retrieval ก็เพิ่ม head ใหม่ใน `models/heads/`
5. ถ้าจะใช้ dataset จริง ให้ต่อจาก `data/image_text_dataset.py` ได้เลย หรือแยก dataset class ใหม่สำหรับ format ของตัวเอง

จุดที่ควรรู้ตอนพัฒนาคือ tensor shape ฝั่งภาพ

- CNN ส่งออกมาเป็น `[B, C, H, W]`
- ใน `utils/tensor_ops.py` จะ flatten เป็น `[B, N, C]` โดยที่ `N = H x W`
- พอจะย้อนกลับเป็น spatial map ก็ใช้ `sequence_to_spatial(...)`

ถ้าจะเปลี่ยน backbone ตัวไหน อย่าลืมเช็กว่า embedding dimension ของ vision, text และ fusion ยังตรงกันอยู่ ไม่งั้น `models/svlb.py` จะ raise error กันไว้ก่อน

## วิธีรัน

### 1. Smoke test

อันนี้ใช้เช็กว่าโมเดลรันได้จริงแบบไม่ต้องมี dataset หรือ checkpoint

```bash
python3 main.py --config configs/default.yaml
```

หรือใช้ shell script

```bash
bash scripts/smoke_test.sh
```

ผลลัพธ์ที่คาดหวังคือจะพิมพ์ shape ของ `vision_tokens`, `text_tokens` และ `spatial_size` ออกมา

### 2. Zero-shot inference

ใช้กับรูปจริงและข้อความจริงได้เลย

ตรง `--image` ต้องใส่ path ของไฟล์ภาพที่มีอยู่จริงในเครื่องนะ ถ้าใส่ `path/to/image.jpg` ตรง ๆ แบบในตัวอย่างโดยไม่เปลี่ยน ระบบจะหาไฟล์ไม่เจอ

```bash
python3 inference.py \
  --config configs/default.yaml \
  --image ./samples/image.jpg \
  --text "a dog running on grass" \
  --text "a white truck in a parking lot"
```

ถ้ามี checkpoint แล้วก็ใส่เพิ่มได้

```bash
python3 inference.py \
  --config configs/default.yaml \
  --image ./samples/image.jpg \
  --checkpoint ./checkpoints/model.pth \
  --text "a dog running on grass" \
  --text "a white truck in a parking lot"
```

หรือใช้ shell script

```bash
bash scripts/run_inference.sh ./samples/image.jpg "a dog running on grass"
```

ถ้ายังไม่มีรูปทดสอบจริง จะลองก็แค่สร้างโฟลเดอร์ `samples/` แล้วเอารูปอะไรก็ได้มาใส่ก่อน เช่น `samples/cat.jpg`

## วิธี test ตอนนี้

ตอนนี้ใน repo ยังไม่มี unit test หรือ integration test แบบเต็ม ๆ นะ มีเป็นการทดสอบเชิงใช้งานอยู่ 2 แบบ

1. `main.py` สำหรับ smoke test
2. `inference.py` สำหรับลองวิ่งกับรูปจริง + ข้อความจริง

ถ้าจะเช็ก syntax ทั้งโปรเจกต์แบบเร็ว ๆ ใช้ได้แบบนี้

```bash
python3 -m compileall .
```

ถ้าจะพัฒนาต่อให้จริงจังขึ้น แนะนำให้เพิ่มชุด test แบบนี้ต่อ

1. test สำหรับ `spatial_to_sequence()` และ `sequence_to_spatial()`
2. test สำหรับ dataset loading และ tokenization
3. test สำหรับ model forward pass ด้วย input ขนาดเล็ก
4. test สำหรับ config validation

## หมายเหตุเล็กน้อย

- ตอนนี้ tokenizer เป็นเวอร์ชันง่าย ๆ เอาไว้ให้ scaffold เดินได้ก่อน ยังไม่ใช่ production tokenizer
- zero-shot ตอนนี้เป็นแนว score ranking จาก head ที่ประกอบไว้ ถ้าจะให้คุณภาพดีจริงควรมี pretrained checkpoint หรือ training pipeline เพิ่ม
- ถ้า `mamba_ssm` ยังไม่พร้อม ระบบจะ fallback ไปใช้ stub เพื่อให้ทดลอง architecture flow ได้ก่อน

## สรุปสั้น ๆ

ถ้าอยากเริ่มเร็วสุด ทำตามนี้ได้เลย

1. ติดตั้ง dependency
2. รัน `python3 main.py --config configs/default.yaml`
3. ลอง `inference.py` กับรูปจริง
4. ค่อยแตกโมดูลเพิ่มตามโจทย์ของงานวิจัย

ถ้าจะพัฒนาต่อเป็น training pipeline, evaluation pipeline หรือเพิ่ม unit tests ก็สามารถต่อจาก scaffold นี้ได้ค่อนข้างตรงไปตรงมา
