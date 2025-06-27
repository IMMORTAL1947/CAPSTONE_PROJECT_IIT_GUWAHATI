import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


st.set_page_config(page_title="Customer Support Assistant", layout="centered")
st.title("💬 AI-Powered Customer Support")
st.write("Ask any question you'd ask a support team!")


@st.cache_resource
def load_model():
    base_model_id = "mistralai/Mistral-7B-Instruct-v0.1"  
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, "finetuned_customer_support_model")
    return model.eval(), tokenizer

model, tokenizer = load_model()


user_query = st.text_input("📝 Enter your customer query here:", placeholder="e.g., How do I return a product?")


if st.button("Generate Response") and user_query:
    with st.spinner("Generating response..."):
        prompt = f"### Customer Query:\n{user_query}\n\n### Response:"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=150)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        
        clean_response = response.split("### Response:")[-1].strip()
        st.success(clean_response)
