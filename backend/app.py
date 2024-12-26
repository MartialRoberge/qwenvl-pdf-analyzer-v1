from flask import Flask, request, jsonify
from flask_cors import CORS
from pdf2image import convert_from_bytes
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, AutoTokenizer
import torch

import matplotlib.pyplot as plt
import io

app = Flask(__name__)
CORS(app)

# Configuration du device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Utilisation de MPS")
else:
    device = torch.device("cpu")
    print("Utilisation CPU")


min_pixels = 256*28*28
max_pixels = 1280*28*28


# Chargement du modèle
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels) #Pas obligé mais je crois que y'a un lien entre le nb de token et la taille des images


#model = Qwen2VLForConditionalGeneration.from_pretrained(
#        "Qwen/Qwen2-VL-2B-Instruct",
#        trust_remote_code=True,
#        torch_dtype=torch.float16)

 #We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2VLForConditionalGeneration.from_pretrained(
     "Qwen/Qwen2-VL-2B-Instruct",
      #attn_implementation="flash_attention_2", #On peut pas l'utiliser sur Mac.... Snif
     torch_dtype=torch.float16,
    trust_remote_code=True
 )

model = model.to(device)


def process_vision_info(messages):
    image_list = []
    for message in messages:
        for content in message["content"]:
            if content["type"] == "image":
                image_list.append(content["image"])
    return image_list, len(image_list)


@app.route('/analyze', methods=['POST'])
def analyze_pdf():
    try:
        if 'pdf' not in request.files:
            return jsonify({'error': 'Aucun fichier PDF fourni'}), 400

        pdf_file = request.files['pdf']
        pdf_bytes = pdf_file.read()

        # Conversion du PDF en images
        images = convert_from_bytes(pdf_bytes)
        full_text = []

        for i, image in enumerate(images):
            print(f"Traitement de la page {i + 1}")

            # Redimensionnement de l'image
            image = image.resize((1024,1024)), # Essayer de respecter le ratio de 3:4  (784,1050)

            #plt.imshow(image)
            #plt.axis('off')  # Pour masquer les axes
            #plt.show()


            # Préparation du message
            #query = """Analyze this financial document and extract the most important information for each section. Provide every table you see from left to right"""  #Please describe what you see in this document in detail.

            query = f"""
            You are analyzing a financial document and currently processing page 1.
            Please follow these instructions strictly:
            1. Identify and extract only the information explicitly present on this page. If an item is missing, write "Not present".
            2. For costs, risk indicators, and performance data, provide exact details from the page without assuming missing information.
            3. Enumerate all tables found and summarize their contents.
            4. Validate all ISIN numbers and identifiers against their standard format.
            """



            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

            # Tokeniser le texte
            tokens = tokenizer(query, return_tensors="pt")

            # Nombre total de tokens textuels
            num_tokens = len(tokens.input_ids[0])

            print(f"Nombre de tokens textuels : {num_tokens}")

            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": query}
            ]}]

            # Application du template de chat
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) #si on apply pas chat template il raconte plus sa vie

            # Obtention des images
            image_inputs, _ = process_vision_info(messages)

            # Préparation des inputs
            inputs = processor(
                text=[text],  # Multiplication pour augmenter le batch
                images=image_inputs,
                padding= True, #"max_length",
                #max_length= 5000,  #1024
                return_tensors="pt"
            )

           # print(f"Dimensions des images : {image_inputs[0].size}")  # Vérifie les dimensions d'une image
           # print(inputs['pixel_values'].shape)  # (batch_size, channels, height, width)

            # Nombre de tokens visuels
           # num_visual_tokens = inputs['pixel_values'].size(2) * inputs['pixel_values'].size(3) // (14 * 14)
           # print(f"Nombre de tokens visuels : {num_visual_tokens}")


            # Déplacement des tenseurs vers le device
            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

            # Génération
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=1000,  # Limite du nombre de nouveaux tokens générés
                    do_sample= True,  # Active l'échantillonnage stochastique (Hasard)
                    #temperature=0.5,  # Réglage de la créativité
                    #top_p= 1,  # Nucleus sampling pour limiter les options improbables
                    #top_k= 40,  # Limitation au top-k tokens les plus probables
                    #repetition_penalty=1.3  # Pour éviter les répétitions excessives

                )

            # Décodage
            output = output.cpu()
            generated_text = processor.batch_decode(output, skip_special_tokens=True)[0]
            full_text.append(f"Page {i + 1}:\n{generated_text}\n")
            print(f"Page {i + 1} traitée avec succès")

        return jsonify({
            'text': '\n'.join(full_text)
        })

    except Exception as e:
        print(f"Erreur lors du traitement: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Erreur lors du traitement: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004, debug=True)