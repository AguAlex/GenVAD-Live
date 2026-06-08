import torch
import os
from PIL import Image, ImageDraw, ImageFilter
from diffusers import AutoPipelineForInpainting

def generare_anomalie_persoana_jos():
    # --- CONFIGURARE ---
    CALE_INPUT = "/content/test.png"          
    CALE_OUTPUT = "/content/rezultat_final_persoana.png"
    
    if not os.path.exists(CALE_INPUT):
        print(f"❌ Imaginea '{CALE_INPUT}' lipsește din director.")
        return

    print("🚀 Se încarcă Juggernaut XL v9...")
    
    pipe = AutoPipelineForInpainting.from_pretrained(
        "RunDiffusion/Juggernaut-XL-v9",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    ).to("cuda")

    pipe.safety_checker = None

    print("🖼️ Se procesează imaginea originală...")
    img_orig = Image.open(CALE_INPUT).convert("RGB")
    w_orig, h_orig = img_orig.size
    
    new_w = 1024
    new_h = int((new_w * (h_orig / w_orig)) // 64) * 64
    img = img_orig.resize((new_w, new_h), Image.LANCZOS)

    print("🎭 Se creează masca...")
    mask = Image.new("L", (new_w, new_h), 0) 
    draw = ImageDraw.Draw(mask)
    
    # ⚠️ ATENȚIE: Trebuie să pui coordonate noi aici!
    # Găsește o zonă liberă pe asfalt. 
    # Deoarece omul este întins, X-urile trebuie să fie depărtate, iar Y-urile apropiate.
    # Exemplu generic (modifică-le uitându-te pe poza originală):
    orig_x1, orig_y1 = 600, 500  
    orig_x2, orig_y2 = 900, 580  

    scale_x = new_w / w_orig
    scale_y = new_h / h_orig

    new_x1 = int(orig_x1 * scale_x)
    new_y1 = int(orig_y1 * scale_y)
    new_x2 = int(orig_x2 * scale_x)
    new_y2 = int(orig_y2 * scale_y)

    rect_coords = [new_x1, new_y1, new_x2, new_y2] 
    
    draw.rectangle(rect_coords, fill=255) 
    mask_blurred = mask.filter(ImageFilter.GaussianBlur(15))

    # 4. Noile Prompt-uri pentru Persoană
    PROMPT = (
        "A low-fidelity, wide-angle CCTV surveillance photo of an industrial lot. "
        "A person wearing casual clothes is lying completely flat on the asphalt ground, motionless. "
        "The person's body casts a realistic shadow on the pavement. "
        "Grainy footage, overcast daylight, desaturated colors, high angle shot."
    )

    NEGATIVE_PROMPT = (
        "fire, smoke, standing, walking, running, portrait, studio lighting, high fashion, "
        "clean, perfect, glowing, bright vibrant colors, cartoon, drawing, CGI, 3d render, "
        "watermark, text, multiple people, floating"
    )

    print("🧍 Se generează anomalia...")
    
    result = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=img,               
        mask_image=mask_blurred, 
        num_inference_steps=50,  
        guidance_scale=6.5,      # Puțin mai mare ca să respecte promptul strict
        strength=0.9,            # ⬅️ FOARTE IMPORTANT: 0.9 pentru a rescrie complet asfaltul
    ).images[0]

    print("💾 Se salvează rezultatele...")
    result.resize((w_orig, h_orig), Image.LANCZOS).save(CALE_OUTPUT)
    mask_blurred.resize((w_orig, h_orig)).save("/content/debug_mask_persoana.png")
    
    print(f"✅ Gata! Verifică {CALE_OUTPUT}")

if __name__ == "__main__":
    generare_anomalie_persoana_jos()