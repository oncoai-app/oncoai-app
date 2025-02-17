# Disease configurations
DISEASE_CONFIGS = {
    "Brain Cancer": {
        "MODEL_URL": "https://huggingface.co/OncoAI/oncobank/resolve/main/oncoai_brain_mri_sartaj_br35h.pth",
        "CATEGORIES": ["Glioma", "Meningioma", "Normal", "Pituitary Tumor"],
        "CONDITION_DESCRIPTIONS": {
            "Glioma": "A malignant tumor that starts in the brain or spine, requiring urgent treatment and care.",
            "Meningioma": "A tumor that forms on the membranes covering the brain and spinal cord, often benign.",
            "Normal": "The MRI scan appears normal, and no abnormalities have been detected.",
            "Pituitary Tumor": "A tumor located in the pituitary gland, which may affect hormone levels, requires medical attention."
        },
        "UPLOAD_TITLE": "Upload MRI Image(s)",
        "CAMERA_TITLE": "Capture MRI Image",
        "SUBTITLE": "Upload or capture an MRI image from the sidebar to analyze potential conditions.",
        "WARNING_MESSAGE": "The AI detected signs of {prediction}. Please consult a neurologist for further evaluation.",
        "INFO_MESSAGE": "Please upload or capture an MRI image from the sidebar to proceed.",
        "SUCCESS_MESSAGE": "The MRI scan appears normal. No abnormalities detected."
    },
    "Skin Cancer": {
        "MODEL_URL": "https://huggingface.co/OncoAI/oncobank/resolve/main/oncoai_skin_photo_isic.pth",
        "CATEGORIES": ["Benign", "Malignant"],
        "CONDITION_DESCRIPTIONS": {
            "Benign": "The lesion appears non-cancerous and typically does not pose a threat to health.",
            "Malignant": "The lesion may be cancerous and requires immediate medical attention."
        },
        "UPLOAD_TITLE": "Upload Skin Lesion Image(s)",
        "CAMERA_TITLE": "Capture Skin Lesion Image",
        "SUBTITLE": "Upload or capture a skin lesion image from the sidebar to analyze potential conditions.",
        "WARNING_MESSAGE": "The AI detected signs of {prediction} growth. Please consult a dermatologist for further evaluation.",
        "INFO_MESSAGE": "Please upload or capture a skin lesion image from the sidebar to proceed.",
        "SUCCESS_MESSAGE": "The skin lesion appears non-cancerous. No immediate concern."
    },
    "Breast Cancer": {
        "MODEL_URL": "https://huggingface.co/OncoAI/oncobank/resolve/main/oncoai_breast_hpe_breakhis.pth",
        "CATEGORIES": ["Benign", "Malignant"],
        "CONDITION_DESCRIPTIONS": {
            "Benign": "The lesion appears non-cancerous and is unlikely to pose a threat to health, but may require routine monitoring.",
            "Malignant": "The lesion may be cancerous and requires immediate medical evaluation and further testing."
        },
        "UPLOAD_TITLE": "Upload Pathology Slide(s)",
        "CAMERA_TITLE": "Capture Pathology Slide Image",
        "SUBTITLE": "Upload or capture a pathology slide image from the sidebar to analyze potential conditions.",
        "WARNING_MESSAGE": "The AI detected signs of {prediction} growth. Please consult an pathologist for further evaluation.",
        "INFO_MESSAGE": "Please upload or capture a pathology slide image from the sidebar to proceed.",
        "SUCCESS_MESSAGE": "The image shows no signs of cancer. There is no immediate concern."
    },
    "Lung Cancer": {
        "MODEL_URL": "https://huggingface.co/OncoAI/oncobank/resolve/main/oncoai_lung_ct_iqothnccd.pth",
        "CATEGORIES": ["Benign", "Malignant"],
        "CONDITION_DESCRIPTIONS": {
            "Benign": "The lesion appears non-cancerous and is unlikely to pose a threat to health, but may require routine monitoring.",
            "Malignant": "The lesion may be cancerous and requires immediate medical evaluation and further testing."
        },
        "UPLOAD_TITLE": "Upload CT Scan(s)",
        "CAMERA_TITLE": "Capture CT Scan",
        "SUBTITLE": "Upload or capture a CT scan from the sidebar to analyze potential conditions.",
        "WARNING_MESSAGE": "The AI detected signs of {prediction} growth. Please consult an oncologist for further evaluation.",
        "INFO_MESSAGE": "Please upload or capture a CT scan from the sidebar to proceed.",
        "SUCCESS_MESSAGE": "The image shows no signs of cancer. There is no immediate concern."
    },
    "Colon Cancer": {
        "MODEL_URL": "https://huggingface.co/OncoAI/oncobank/resolve/main/oncoai_colon_hpe_lc25000.pth",
        "CATEGORIES": ["Benign", "Malignant"],
        "CONDITION_DESCRIPTIONS": {
            "Benign": "The lesion appears non-cancerous and is unlikely to pose a threat to health, but may require routine monitoring.",
            "Malignant": "The lesion may be cancerous and requires immediate medical evaluation and further testing."
        },
        "UPLOAD_TITLE": "Upload Pathology Slide(s)",
        "CAMERA_TITLE": "Capture Pathology Slide Image",
        "SUBTITLE": "Upload or capture a pathology slide image from the sidebar to analyze potential conditions.",
        "WARNING_MESSAGE": "The AI detected signs of {prediction} growth. Please consult an pathologist for further evaluation.",
        "INFO_MESSAGE": "Please upload or capture a pathology slide image from the sidebar to proceed.",
        "SUCCESS_MESSAGE": "The image shows no signs of cancer. There is no immediate concern."
    },
    "Osteosarcoma": {
        "MODEL_URL": "https://huggingface.co/OncoAI/oncobank/resolve/main/oncoai_osteo_hpe_sarcoma.pth",
        "CATEGORIES": ["Non-Tumor", "Non-Viable Tumor", "Viable Tumor"],
        "CONDITION_DESCRIPTIONS": {
            "Non-Tumor": "The lesion appears non-cancerous and is unlikely to pose a threat to health, but may require routine monitoring.",
            "Non-Viable Tumor": "The tumor appears non-viable, suggesting that it may not be actively growing or threatening health at this time.",
            "Viable Tumor": "The tumor appears to be viable and could be cancerous, requiring immediate medical evaluation and further testing."
        },
        "UPLOAD_TITLE": "Upload Pathology Slide(s)",
        "CAMERA_TITLE": "Capture Pathology Slide Image",
        "SUBTITLE": "Upload or capture a pathology slide image from the sidebar to analyze potential conditions.",
        "WARNING_MESSAGE": "The AI detected signs of a {prediction}. Please consult an oncologist or pathologist for further evaluation.",
        "INFO_MESSAGE": "Please upload or capture a pathology slide image from the sidebar to proceed.",
        "SUCCESS_MESSAGE": "There appears to be no tumor in the pathology slide presented."
    },
    "Ocular Neoplasm": {
        "MODEL_URL": "https://huggingface.co/OncoAI/oncobank/resolve/main/oncoai_neoplasm_fundus_jsiec.pth",
        "CATEGORIES": ["Neoplasm", "Normal"],
        "CONDITION_DESCRIPTIONS": {
            "Neoplasm": "The image suggests the presence of a tumor or growth in the eye. This could indicate a benign or malignant condition, requiring further evaluation by a medical professional.",
            "Normal": "The image shows no signs of any ocular abnormalities. The eye appears healthy, with no indication of tumors or other concerning conditions."
        },
        "UPLOAD_TITLE": "Upload Fundus Photograph(s)",
        "CAMERA_TITLE": "Capture Eye Image",
        "SUBTITLE": "Upload or capture a fundus image from the sidebar to analyze potential conditions.",
        "WARNING_MESSAGE": "The AI detected signs of a {prediction}. Please consult an ophthalmologist for further evaluation.",
        "INFO_MESSAGE": "Please upload or capture a fundus image from the sidebar to proceed.",
        "SUCCESS_MESSAGE": "There appears to be no ocular neoplasm in the image provided."
    }
}

# Define COLORS for predictions
COLORS = {
    "Benign": "#4CAF50",  # Green
    "Malignant": "#F44336",  # Red
    "Glioma": "#FF5722",  # Orange
    "Meningioma": "#2196F3",  # Blue
    "Normal": "#4CAF50",  # Green
    "Pituitary Tumor": "#FFEB3B",  # Yellow
    "Non-Tumor": "#4CAF50",  # Green
    "Non-Viable Tumor": "#FFEB3B",  # Yellow
    "Viable Tumor": "#F44336",  # Red
    "Neoplasm": "#FF5722",  # Orange
}
