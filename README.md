# Welcome to the Project

Please ensure the following guidelines are followed throughout the course of this project:

## 🎥 Demo

The red-colored human is the input to the motion prediction model, and the blue-colored human is the generated output.

- 🔹 **Ground Truth**: [![Ground Truth](https://drive.google.com/uc?id=1Z2l2_zCwFvXxFsr3f6JcJ_N0hViWOjgL)](https://drive.google.com/file/d/1Z2l2_zCwFvXxFsr3f6JcJ_N0hViWOjgL/view)
- 🔹 **Predicted**: [![Predicted](https://drive.google.com/uc?id=1xR0riCAV4j2YPRvEj4kZpQOPXL-PpwfV)](https://drive.google.com/file/d/1xR0riCAV4j2YPRvEj4kZpQOPXL-PpwfV/view?usp=sharing)

---

## 🔄 Code Management

- **The true source of code is this repository** — always make sure it is kept up to date.

## ☁️ Data Management

- **The true source of data (files listed in `.gitignore`) resides in the corresponding GCS (Google Cloud Storage) buckets.**
- Always keep the GCS buckets synchronized and updated with the local data.

---

## 📂 Mapping: `.gitignore` Folders to GCS Buckets

| Local Folder Path                                    | GCS Bucket Path                              |
|------------------------------------------------------|-----------------------------------------------|
| `Human_Motion_Refine (1)/data/`                     | `gs://arun-ml-bucket/data/`                  |
| `Human_Motion_Refine (1)/smplx/`                    | `gs://arun-ml-bucket/smplx/`                 |
| `Code/FNO/burgers_data_R10.mat`                     | `gs://arun-ml-bucket/FNO-Demo-Data/`         |
| `Code/Duolando-main/assets/`                        | `gs://arun-ml-bucket/assets/`                |
| `Code/Duolando-main/data/`                          | `gs://arun-ml-bucket/main-data/`             |
| `Code/Duolando-main/data_more_processing/`          | `gs://arun-ml-bucket/data_more_processing/`  |
| `Code/Duolando-main/new_data/`                      | `gs://arun-ml-bucket/new_data/`              |
| `Code/Duolando-main/smplx/`                         | `gs://arun-ml-bucket/main-smplx/`            |
| `Code/Duolando-main/misc-data-files/`               | `gs://arun-ml-bucket/misc-data-files/`       |

---

Please reach out to the project maintainer if you have any questions or need access to the buckets.
