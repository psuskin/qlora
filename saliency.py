import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import captum
import numpy as np
from peft import PeftModel

import pandas as pd
import seaborn as sns

def load_model(model_name_or_path='huggyllama/llama-7b', adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # Fixing some of the early LLaMA HF conversion issues.
    tokenizer.bos_token_id = 1

    # Load the model (use bf16 for faster inference)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )
    )

    if adapter_path:
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
    else:
        model = base_model

    return model, tokenizer

model, tokenizer = load_model('meta-llama/Llama-2-13b-hf', 'output/klio-alpaca-2-13b-r64-noeval/checkpoint-1875/adapter_model')

forward = lambda input_embedding, model, input_attention_mask, target_id: model(input_ids=None,
                                                                                inputs_embeds=input_embedding,
                                                                                attention_mask=input_attention_mask).logits[
                                                                          :, -1, :].squeeze(1).to(model.device).softmax(dim=-1).gather(
    -1, target_id.reshape(-1, 1)).squeeze(-1)
method = captum.attr.InputXGradient(forward_func=forward)


def getOutput(text):
    inputs = tokenizer.encode(text, return_tensors="pt")

    outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=2048, return_dict_in_generate=True)
    target_ids = outputs.sequences.to(model.device)

    target_text = tokenizer.decode(target_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return (target_ids, target_text), (inputs.shape[1], target_ids.shape[1])


def getEmbeddings(target_ids, target_text):
    batch = tokenizer(
        text=target_text,
        text_target=target_text,
        add_special_tokens=True,
        #padding=True,
        #truncation=True,
        return_tensors="pt",
    )
    attention_mask = batch["attention_mask"]
    input_embeddings = model.get_input_embeddings()(target_ids).to(model.device)

    return attention_mask, input_embeddings


def getAttributionOutputs(target_ids, attention_mask, input_embeddings, inputLength, targetLength):
    attribution_outputs = []
    for step in range(inputLength, targetLength):
        target_id = target_ids[0, step].unsqueeze(0)

        input_embedding = input_embeddings[:, :step]
        input_attention_mask = attention_mask[0, :step].unsqueeze(0)

        with torch.no_grad():
            output = model(input_ids=None, inputs_embeds=input_embedding, attention_mask=input_attention_mask,
                           output_attentions=False, output_hidden_states=False)
        logits = output.logits[:, -1, :].squeeze(1).to(model.device)
        step_scores = {
            "probability": logits.softmax(dim=-1).gather(-1, target_id.reshape(logits.shape[0], 1)).squeeze(-1)}

        attr = method.attribute(inputs=(input_embedding,),
                                additional_forward_args=(model, input_attention_mask, target_id))
        target_attributions = attr[0].detach()

        target_token = tokenizer.decode(target_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        attribution_outputs.append((target_token, target_attributions, step_scores))

    return attribution_outputs


def consolidateAttributionOutputs(attribution_outputs, inputLength, targetLength):
    bsteps = [attribution_outputs[i][1] for i in range(len(attribution_outputs))]
    dim_ranges = {dim: [bstep.shape[dim] for bstep in bsteps] for dim in range(bsteps[0].ndim)}
    for dim, dim_range in dim_ranges.items():
        # If dimension grows across batch steps, it will be padded
        if max(dim_range) > min(dim_range):
            for bstep_idx, bstep in enumerate(bsteps):
                padded_bstep = torch.ones(
                    *bstep.shape[:dim],
                    max(dim_range) - bstep.shape[dim],
                    *bstep.shape[dim + 1:],  # noqa
                    dtype=bstep.dtype,
                    device=bstep.device,
                )
                padded_bstep = torch.cat([bstep, padded_bstep * float("nan")], dim=dim)
                bsteps[bstep_idx] = padded_bstep
    dim = 2 if bsteps[0].ndim > 1 else 1
    sequences = torch.stack(bsteps, dim=dim)
    sequences = sequences.split(1, dim=0)
    squeezed_sequences = [seq.squeeze(0) for seq in sequences]
    target_attributions = squeezed_sequences[0]
    start_idx = 0
    end_idx = targetLength
    target_attributions = target_attributions[
                          start_idx:end_idx, : (end_idx - inputLength), ...  # noqa: E203
                          ]
    # if target_attributions.shape[0] != end_idx:
    #     empty_final_row = torch.ones(1, *target_attributions.shape[1:]) * float("nan")
    #     target_attributions = torch.cat([target_attributions, empty_final_row], dim=0)

    tokens = [attribution_outputs[i][0] for i in range(len(attribution_outputs))]

    probabilities = torch.tensor([attribution_outputs[i][2]["probability"] for i in range(len(attribution_outputs))])

    attributions = torch.linalg.vector_norm(target_attributions, ord=2, dim=-1)
    if False:
        nan_mask = attributions.isnan()
        attributions[nan_mask] = 0.0
        attributions = torch.nn.functional.normalize(attributions, p=1, dim=0)
        attributions[nan_mask] = float("nan")

    attributed = []
    for i in range(len(tokens)):
        attributed.append((tokens[i], attributions[:, i].detach().cpu().float().numpy(), probabilities[i].item()))

    return attributed


if __name__ == "__main__":
    (target_ids, target_text), (inputLength, targetLength) = getOutput('Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nWhich attributes exist? Context: This is the description of the module "attribut" with the name "Attribute (module)": There are three attribute types in ClassiX®: Preset material characteristic Calculated material characteristic Conditional material characteristic You can find more information in the topic Features. This is the description of the functionality of the module "attribut" with the name "Attribute (module)" regarding Input window: This window is used to maintain the attributes. It varies for the three attribute types, but behaves almost the same. Note: Characteristics are clearly defined via the data field. Therefore, each data field should only be used once, otherwise unwanted results may occur when integrating the attributes within the quotation and order items. This is the description of the functionality of the module "attribut" with the name "Attribute (module)" regarding List window: Serves to list the attribute objects. This is the description of the functionality of the module "attribut" with the name "Attribute (module)" regarding Selection window: This window is used to select an attribute object.\n\n### Response:')
    subwords = tokenizer.convert_ids_to_tokens(target_ids[0].tolist())
    attention_mask, input_embeddings = getEmbeddings(target_ids, target_text)
    attribution_outputs = getAttributionOutputs(target_ids, attention_mask, input_embeddings, inputLength, targetLength)
    attributed = consolidateAttributionOutputs(attribution_outputs, inputLength, targetLength)
    # print(subwords, attributed)

    tokens, arrays, probabilities = zip(*attributed)

    df = pd.DataFrame(np.asarray(arrays).T)
    df.loc[len(df.index)] = probabilities

    cmap = sns.light_palette("red", as_cmap=True)

    styled_df_full = df.style.format("{:.3f}").background_gradient(axis=None, subset=pd.IndexSlice[:len(
        df.index) - 2]).background_gradient(axis="columns", cmap=cmap, subset=pd.IndexSlice[len(df.index) - 1:])
    styled_df_full.relabel_index(subwords[:-1] + ["probability"], axis="index")
    styled_df_full.relabel_index(tokens, axis="columns")

    styled_df_column = df.style.background_gradient(subset=pd.IndexSlice[:len(df.index) - 2]).background_gradient(
        axis="columns", cmap=cmap, subset=pd.IndexSlice[len(df.index) - 1:])
    styled_df_column.relabel_index(subwords[:-1] + ["probability"], axis="index")
    styled_df_column.relabel_index(tokens, axis="columns")

    with open("saliency.html", "w", encoding="utf-8") as f:
        f.write(styled_df_column.to_html())