# Dataset Structure

This document describes the required dataset organization for SIREN toolkit evaluation.

## DCASE Challenge Series (2020-2025)

### DCASE 2020 (Task 2)
```text
{dataset_root}/dcase2020_t2/
├── development/
│   └── {fan,pump,slider,ToyCar,ToyConveyor,valve}/
│       ├── train/normal/*.wav
│       └── test/{normal,anomaly}_id_XX_*.wav
└── evaluation/
    ├── {fan,pump,slider,ToyCar,ToyConveyor,valve}/test/id_XX_*.wav
    └── eval_data_list_2020_converted.csv   # test_filename,reference_filename,label
```

### DCASE 2021 (Task 2)
```text
{dataset_root}/dcase2021_t2/
├── development/
│   └── {fan,pump,slider,gearbox,ToyCar,ToyTrain,valve}/
│       └── section_{XX}_{source|target}_test_{anomaly|normal}_*.wav
└── evaluation/
    ├── {fan,pump,slider,gearbox,ToyCar,ToyTrain,valve}/section_{XX}_*.wav
    └── eval_data_list_2021_converted.csv
```

### DCASE 2022 (Task 2)
```text
{dataset_root}/dcase2022_t2/
├── development/{machine}/section_{XX}_{source|target}_test_{anomaly|normal}_{####}_{cond}.wav
└── evaluation/
    ├── {machine}/section_{XX}_*.wav
    └── eval_data_list_2022_converted.csv
```

### DCASE 2023 (Task 2)
```text
{dataset_root}/dcase2023_t2/
├── development/{machine}/section_{XX}_{source|target}_test_{anomaly|normal}_{####}_{cond}.wav
└── evaluation/
    ├── {machine}/*_eval_*.wav
    └── eval_data_list_2023_converted.csv
```

### DCASE 2024 (Task 2)
```text
{dataset_root}/dcase2024_t2/
├── development/{machine}/...
└── evaluation/
    ├── {machine}/*_eval_*.wav
    └── eval_data_list_2024_converted.csv
```

### DCASE 2025 (Task 2)
```text
{dataset_root}/dcase2025_t2/
├── development/{machine}/...
└── evaluation/
    ├── {machine}/*_eval_*.wav
    └── eval_data_list_2025_converted.csv
```

### Evaluation Label Files
- Filename: `eval_data_list_YYYY_converted.csv`
- Location: place under the corresponding year's `evaluation/` directory
- Columns: `test_filename,reference_filename,label` (label in {0,1})
- Convenience: copies for 2020–2025 are also included in this toolkit at `siren/supplements/eval_lists/`

## Fault Diagnosis & Classification Datasets

### MAFAULDA (Multi-fault Bearing Dataset)
```
/path/to/MAFAULDA/
├── horizontal-misalignment/
│   ├── 0.5mm/
│   ├── 1.0mm/
│   ├── 1.5mm/
│   └── 2.0mm/
├── vertical-misalignment/
│   ├── 0.51mm/
│   ├── 0.63mm/
│   ├── 1.27mm/
│   ├── 1.40mm/
│   ├── 1.78mm/
│   └── 1.90mm/
├── imbalance/
│   ├── 6g/
│   ├── 10g/
│   ├── 15g/
│   ├── 20g/
│   ├── 25g/
│   ├── 30g/
│   └── 35g/
├── overhang/
│   ├── ball_fault/
│   ├── cage_fault/
│   └── outer_race/
├── underhang/
│   ├── ball_fault/
│   ├── cage_fault/
│   └── outer_race/
└── normal/
```

### CWRU (Case Western Reserve University Bearing Dataset)
```
/path/to/CWRU/
├── 12k_Drive_End_Bearing_Fault_Data/
│   ├── B/
│   │   ├── 007/
│   │   ├── 014/
│   │   ├── 021/
│   │   └── 028/
│   ├── IR/
│   │   ├── 007/
│   │   ├── 014/
│   │   ├── 021/
│   │   └── 028/
│   └── OR/
│       ├── 007/
│       ├── 014/
│       └── 021/
├── 12k_Fan_End_Bearing_Fault_Data/
│   ├── B/
│   │   ├── 007/
│   │   ├── 014/
│   │   └── 021/
│   ├── IR/
│   │   ├── 007/
│   │   ├── 014/
│   │   └── 021/
│   └── OR/
│       ├── 007/
│       ├── 014/
│       └── 021/
├── 48k_Drive_End_Bearing_Fault_Data/
│   ├── B/
│   │   ├── 007/
│   │   ├── 014/
│   │   └── 021/
│   ├── IR/
│   │   ├── 007/
│   │   ├── 014/
│   │   └── 021/
│   └── OR/
│       ├── 007/
│       ├── 014/
│       └── 021/
└── Normal/
```

### IIEE (IDMT-ISA Electric Engine Dataset)
```
/path/to/IDMT-ISA-Electric-Engine/
├── train/
│   ├── engine1_good/pure.wav
│   ├── engine2_broken/pure.wav
│   └── engine3_heavyload/pure.wav
├── train_cut/
│   ├── engine1_good/pure_*.wav   # 3-second clips, 44.1 kHz, mono, 16-bit
│   ├── engine2_broken/pure_*.wav
│   └── engine3_heavyload/pure_*.wav
└── test/
    ├── engine1_good/{talking_*.wav, atmo_*.wav, whitenoise_low.wav, stresstest.wav}
    ├── engine2_broken/{...}
    └── engine3_heavyload/{...}
```

### IICA (IDMT-ISA Compressed Air Dataset)
```
/path/to/IDMT-ISA-Compressed-Air/
└── raw/
    ├── tubeleak/{hydr, hydr_low, lab, work, work_low}/{1,2,3}/*.wav  # 48 kHz, mono, 24-bit
    └── ventleak/{hydr, hydr_low, lab, work, work_low}/{1,2,3}/*.wav
```

## Notes

- All audio files should be in WAV format
- Ensure sample rates match the expected values for each dataset
- For DCASE datasets, the evaluation label files are crucial for proper scoring
- Some datasets may require preprocessing or resampling to match expected formats
