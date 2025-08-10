# 🕳️ DeepCave

Offline mobile and desktop application for identifying caves from entrance 
images using DINOv2 embeddings and FAISS similarity search.


## 👥 The Team 
**Team Members**
- Noam Ansbacher — Noam.Ansbacher@mail.huji.ac.il
- Nadav Hardof — Nadav.Hardof@mail.huji.ac.il

**Mentor**
- Matan Levy — levy@cs.huji.ac.il

## 📚 Project Description
DeepCave is a computer vision project designed to identify caves based on their 
entrance images.  

The system uses **DINOv2** embeddings and **FAISS** for efficient similarity 
search, allowing users to match a new cave photo against a local dataset.

**Main features:**
- Offline operation — no internet connection required after setup
- Cave identification from a single photo of the entrance
- Add new cave images to the local database
- Configurable top-K search results
- Works on both desktop and mobile

**Main components:**
- Image preprocessing and feature extraction with DINOv2
- Embedding storage and retrieval with FAISS
- User interface for image input and result display
- classifier training (Linear Head)

**Technologies:**
- Python (PyTorch, FAISS)
- DINOv2 (Meta AI)
- Android Studio (mobile version)
- Kotlin
- JSON for metadata storage

## ⚡ Getting Started

### 🧱 Prerequisites
- Python 3.9+
- pip
- GPU with CUDA support (recommended)
- [PyTorch](https://pytorch.org/)
- [FAISS](https://github.com/facebookresearch/faiss)

### 🏗️ Installing
Clone the repository:
bash:
git clone https://github.com/speedofsound98/DeepCaveProject.git
cd DeepCaveProject
