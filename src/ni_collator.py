import logging
import random
import string
from transformers.data.data_collator import *
from transformers import (
    # OPTPreTrainedModel,
    GPT2PreTrainedModel
)
logger = logging.getLogger(__name__)
import numpy as np
import torch

@dataclass
class DataCollatorForNI:

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    add_task_name: bool = False
    add_task_definition: bool = True
    num_pos_examples: int = 0
    num_neg_examples: int = 0
    add_explanation: bool = False
    tk_instruct: bool = False
    text_only: bool=False


    def __call__(self, batch, return_tensors=None):

        if return_tensors is None:
            return_tensors = self.return_tensors

        sources = []
        extra_model_inputs = {}
        for instance in batch:
            if self.tk_instruct:
                all_valid_encodings = [
                    # instruction only
                    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 0, "num_neg_examples": 0, "add_explanation": False}, 
                    # example only
                    {"add_task_name": False, "add_task_definition": False, "num_pos_examples": 2, "num_neg_examples": 0, "add_explanation": False}, 
                    # instruction + pos examples
                    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 2, "num_neg_examples": 0, "add_explanation": False}, 
                    # instruction + pos examples + neg examples 
                    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 2, "num_neg_examples": 2, "add_explanation": False},
                    # instruction + pos (w. explanation) 
                    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 2, "num_neg_examples": 0, "add_explanation": True}, 
                ]
                encoding_schema = random.choice(all_valid_encodings)
                add_task_name = encoding_schema["add_task_name"]
                add_task_definition = encoding_schema["add_task_definition"]
                num_pos_examples = encoding_schema["num_pos_examples"]
                num_neg_examples = encoding_schema["num_neg_examples"]
                add_explanation = encoding_schema["add_explanation"]
            else:
                add_task_name = self.add_task_name
                add_task_definition = self.add_task_definition
                num_pos_examples = self.num_pos_examples
                num_neg_examples = self.num_neg_examples
                add_explanation = self.add_explanation 

            task_input = ""
            # add the input first.
            task_input += "Now complete the following example -\n"
            task_input += f"Input: {instance['Instance']['input'].strip()}"
            if not task_input[-1] in string.punctuation:
                task_input += "."
            task_input += "\n"
            task_input += "Output: "
            
            task_name = ""
            if add_task_name:
                task_name += instance["Task"] + ". "

            definition = ""
            if add_task_definition:
                if isinstance(instance["Definition"], list):
                    definition = "Definition: " + instance["Definition"][0].strip() # TODO: should we use <Definition>?
                else:
                    definition = "Definition: " + instance["Definition"].strip()
                if not definition[-1] in string.punctuation:
                    definition += "."
                definition += "\n\n"
            
            if isinstance(instance["Definition"], list):
                extra_model_inputs["task_definition"] = "Definition: " + instance["Definition"][0].strip() # TODO: should we use <Definition>?
            else:
                extra_model_inputs["task_definition"] = "Definition: " + instance["Definition"].strip()

            # try to add positive examples.
            pos_examples = []
            for idx, pos_example in enumerate(instance["Positive Examples"][:num_pos_examples]):
                pos_example_str = f" Positive Example {idx+1} -\n"
                pos_example_str += f"Input: {pos_example['input'].strip()}"
                if not pos_example_str[-1] in string.punctuation:
                    pos_example_str += "."
                pos_example_str += "\n"
                pos_example_str += f" Output: {pos_example['output'].strip()}"
                if not pos_example_str[-1] in string.punctuation:
                    pos_example_str += "."
                pos_example_str += "\n" 
                if add_explanation and "explanation" in pos_example:
                    pos_example_str += f" Explanation: {pos_example['explanation'].strip()}"
                    if not pos_example_str[-1] in string.punctuation:
                        pos_example_str += "."
                    pos_example_str += "\n"
                pos_example_str += "\n"
                if len(self.tokenizer(definition + " ".join(pos_examples) + pos_example_str + task_input)["input_ids"]) <= self.max_source_length:
                    pos_examples.append(pos_example_str)
                else:
                    break
            
            # try to add negative examples.
            neg_examples = []
            for idx, neg_example in enumerate(instance["Negative Examples"][:num_neg_examples]):
                neg_example_str = f" Negative Example {idx+1} -\n"
                neg_example_str += f"Input: {neg_example['input'].strip()}"
                if not neg_example_str[-1] in string.punctuation:
                    neg_example_str += "."
                neg_example_str += "\n"
                neg_example_str += f" Output: {neg_example['output'].strip()}"
                if not neg_example_str[-1] in string.punctuation:
                    neg_example_str += "."
                neg_example_str += "\n"
                if add_explanation and "explanation" in neg_example:
                    neg_example_str += f" Explanation: {neg_example['explanation'].strip()}"
                    if not neg_example_str[-1] in string.punctuation:
                        neg_example_str += "."
                    neg_example_str += "\n"
                neg_example_str += "\n"
                if len(self.tokenizer(definition + " ".join(pos_examples) + " ".join(neg_examples) + neg_example_str + task_input)["input_ids"]) <= self.max_source_length:
                    neg_examples.append(neg_example_str)
                else:
                    break 
            
            source = task_name + definition + "".join(pos_examples) + "".join(neg_examples) + task_input
            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.max_source_length:
                sources.append(source)
            else:
                sources.append(self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True))

        model_inputs = {}
        

        # 1. prepare labels first in str format
        if "output" in batch[0]["Instance"] and batch[0]["Instance"]["output"]:
            # Randomly select one reference if multiple are provided.
            labels = [random.choice(ex["Instance"]["output"]) for ex in batch]
            # if self.text_only:
            #     model_inputs["labels"] = labels
            # else:
            #     with self.tokenizer.as_target_tokenizer():
            #         labels = self.tokenizer(
            #             labels,
            #             max_length=self.max_target_length,
            #             padding=self.padding,
            #             return_tensors=self.return_tensors,
            #             truncation=True,
            #             pad_to_multiple_of=self.pad_to_multiple_of
            #         )
            #     label_mask = labels["attention_mask"].bool()
            #     model_inputs["labels"] = labels["input_ids"].masked_fill(~label_mask, self.label_pad_token_id)
        else:
            labels = None
            # model_inputs["labels"] = None
        is_causal_lm = False
        # is_causal_lm =  isinstance(self.model, OPTPreTrainedModel) or isinstance(self.model, GPT2PreTrainedModel)
        if is_causal_lm and labels:
            sources = ["".join(sl) for sl in zip(sources, labels)]

        # 2. prepare model inputs first
        if self.text_only:
            model_inputs = {"inputs": sources}
        else:
            model_inputs = self.tokenizer(
                    sources, 
                    max_length=self.max_source_length, 
                    padding=self.padding,
                    return_tensors=self.return_tensors, 
                    truncation=True,
                    pad_to_multiple_of=self.pad_to_multiple_of)
            # if is_causal_lm:
            #     model_inputs = self.tokenizer(
            #         sources,
            #         max_length=self.max_source_length,
            #         padding=self.padding,
            #         return_tensors=self.return_tensors, 
            #         truncation=True,
            #         pad_to_multiple_of=self.pad_to_multiple_of,
            #         return_overflowing_tokens=True,
            #         return_length=True)
            # else:
            #     model_inputs = self.tokenizer(
            #         sources, 
            #         max_length=self.max_source_length, 
            #         padding=self.padding,
            #         return_tensors=self.return_tensors, 
            #         truncation=True,
            #         pad_to_multiple_of=self.pad_to_multiple_of)

        # 3. prepare model labels second
        if labels and not is_causal_lm:
            if self.text_only:
                model_inputs["labels"] = labels
            else:
                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(
                        labels,
                        max_length=self.max_target_length,
                        padding=self.padding,
                        return_tensors=self.return_tensors,
                        truncation=True,
                        pad_to_multiple_of=self.pad_to_multiple_of
                    )
                label_mask = labels["attention_mask"].bool()
                model_inputs["labels"] = labels["input_ids"].masked_fill(~label_mask, self.label_pad_token_id)
        if is_causal_lm:
            if self.text_only:
                model_inputs["labels"] = model_inputs["inputs"]
            else:
                model_inputs["labels"] = model_inputs["input_ids"]


        # prepare decoder_input_ids
        if self.model is not None and not self.text_only and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])
            model_inputs["decoder_input_ids"] = decoder_input_ids
            # elif hasattr(self.model, "prepare_inputs_for_generation"): # GPT or OPT model
            #     # TODO: delete this
                
            #     model_inputs["inputs"] +=\
            #          f" Output: {model_inputs.pop('labels')}"

        return model_inputs, extra_model_inputs

@dataclass
class DataCollatorForNIDenoising:
    # NI related arguments
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    add_task_name: bool = False
    add_task_definition: bool = True
    num_pos_examples: int = 0
    num_neg_examples: int = 0
    add_explanation: bool = False
    tk_instruct: bool = False
    text_only: bool=False
    # noise span related args
    noise_density: float = 0.15
    mean_noise_span_length: float = 3.0
    decoder_start_token_id: int = 0
    denoise_obj: str = "definition"
    """
    add_task_name: whether to add task_name to denoise objective
    add_task_definition: whether to add task_definition to denoise objective
    num_pos_examples: number of positive examples to use in the input for denoise objective
    num_neg_examples: number of negative examples to use in the input for denoise objective
    add_explanation: whether to add explanation to denoise objective
    denoise_obj: denoise objective, it's either definition or explanation
    """
    # non-lm-collator seq2seq __call__ function
    # return dictionary contains ["labels"] ["decoder_input_ids"]
    
    # lm-collator inherits DataCollatorMixin
    # __call__ funciton https://github.com/huggingface/transformers/blob/f0d496828d3da3bf1e3c8fbed394d7847e839fa6/src/transformers/data/data_collator.py#L35
    # torch.call(features)

    # T5 collator
    # https://github.com/huggingface/transformers/blob/9129fd0377e4d46cb2d0ea28dc1eb91a15f65b77/examples/flax/language-modeling/run_t5_mlm_flax.py#L297

    """
        data_collator = FlaxDataCollatorForT5MLM(
        tokenizer=tokenizer,
        noise_density=data_args.mlm_probability,
        mean_noise_span_length=data_args.mean_noise_span_length,
        input_length=max_seq_length,
        target_length=targets_length,
        pad_token_id=model.config.pad_token_id,
        decoder_start_token_id=model.config.decoder_start_token_id,

    )
    """
    def __call__(self, batch, return_tensors=None):

        # batch[0].keys()
        # dict_keys(['id', 'Task', 'Contributors', 'Source', 'URL', 'Categories', 'Reasoning', 'Definition', 'Positive Examples', 'Negative Examples', 'Input_language', 'Output_language', 'Instruction_language', 'Domains', 'Instance'])
        """
        how to use data collator
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer)
        masked_tokenized_input = self.data_collator(tokenized_input['input_ids'])
        """

        """
        pseudo code for data collator

        # training task wise
        1. prepare model condition inputs (pos examples) first.
        2. prepare model condition inputs (def, explanation) second and corrupt them. Corrupt them by using function directly on those part of string?
        
        # encoding wise
        Gather tokens first
        encode into token ids

        # combined
        # should I encode and masking them separately or together with indexing?
        # I think we should mask them together with indexing. As conditioning inputs has varied length.
        # 1. record special token ["Definition"] ["Explanation"] ids
        # 2. incorporate mask whole tokens and sentence order shuffle etc to masking
        # 3. we could insert in-batch negative definition

        masked_tokenized_input["input_ids"], ["labels"]

        """
        if return_tensors is None:
            return_tensors = self.return_tensors
        if self.denoise_obj == "explanation":
            assert self.num_pos_examples == 1 or self.num_neg_examples == 1, "current implementation only supports one positive/negative example"
        sources = []
        for instance in batch:
            add_task_name = self.add_task_name
            add_task_definition = self.add_task_definition
            assert add_task_definition == True, "add_task_definition is required for NIDenoising"
            num_pos_examples = self.num_pos_examples
            num_neg_examples = self.num_neg_examples
            add_explanation = self.add_explanation

            # task_input = ""
            # # add the input first.
            # task_input += "Now complete the following example -\n"
            # task_input += f"Input: {instance['Instance']['input'].strip()}"
            # if not task_input[-1] in string.punctuation:
            #     task_input += "."
            # task_input += "\n"
            # task_input += "Output: "
            
            # initialize the NI task elements
            task_name = ""
            if add_task_name:
                task_name += instance["Task"] + ". "

            definition = ""
            if add_task_definition:
                if isinstance(instance["Definition"], list):
                    definition = "Definition: " + instance["Definition"][0].strip() # TODO: should we use <Definition>?
                else:
                    definition = "Definition: " + instance["Definition"].strip()
                if not definition[-1] in string.punctuation:
                    definition += "."
                definition += "\n\n"


            # try to add positive examples.
            pos_examples = []
            for idx, pos_example in enumerate(instance["Positive Examples"][:num_pos_examples]):
                pos_example_str = f" Positive Example {idx+1} -\n"
                pos_example_str += f"Input: {pos_example['input'].strip()}"
                if not pos_example_str[-1] in string.punctuation:
                    pos_example_str += "."
                pos_example_str += "\n"
                pos_example_str += f" Output: {pos_example['output'].strip()}"
                if not pos_example_str[-1] in string.punctuation:
                    pos_example_str += "."
                pos_example_str += "\n" 
                if add_explanation and "explanation" in pos_example:
                    pos_example_str += f" Explanation: {pos_example['explanation'].strip()}"
                    if not pos_example_str[-1] in string.punctuation:
                        pos_example_str += "."
                    pos_example_str += "\n"
                pos_example_str += "\n"
                if len(self.tokenizer(definition + " ".join(pos_examples) + pos_example_str)["input_ids"]) <= self.max_source_length:
                    pos_examples.append(pos_example_str)
                else:
                    pos_examples.append(pos_example_str)
                    break
            
            # try to add negative examples.
            neg_examples = []
            for idx, neg_example in enumerate(instance["Negative Examples"][:num_neg_examples]):
                neg_example_str = f" Negative Example {idx+1} -\n"
                neg_example_str += f"Input: {neg_example['input'].strip()}"
                if not neg_example_str[-1] in string.punctuation:
                    neg_example_str += "."
                neg_example_str += "\n"
                neg_example_str += f" Output: {neg_example['output'].strip()}"
                if not neg_example_str[-1] in string.punctuation:
                    neg_example_str += "."
                neg_example_str += "\n"
                if add_explanation and "explanation" in neg_example:
                    neg_example_str += f" Explanation: {neg_example['explanation'].strip()}"
                    if not neg_example_str[-1] in string.punctuation:
                        neg_example_str += "."
                    neg_example_str += "\n"
                neg_example_str += "\n"
                if len(self.tokenizer(definition + " ".join(pos_examples) + " ".join(neg_examples) + neg_example_str)["input_ids"]) <= self.max_source_length:
                    neg_examples.append(neg_example_str)
                else:
                    break 
            if len(pos_examples) == 0:
                raise ValueError("No positive examples found for task: {}".format(instance["Task"]))
            # TODO: denoise explanation
            if self.denoise_obj == "definition":
                source =  "".join(pos_examples) + "".join(neg_examples) + task_name + definition
            elif self.denoise_obj == "explanation":
                source =  task_name + definition + "".join(pos_examples) + "".join(neg_examples)
            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.max_source_length:
                sources.append(source)
            else:
                sources.append(self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True))
        
        # here we have prepared all the sources.
        
        model_inputs = {}
        

        # 1. prepare labels first in str format
        # if "output" in batch[0]["Instance"] and batch[0]["Instance"]["output"]:
        #     # Randomly select one reference if multiple are provided.
        #     labels = [random.choice(ex["Instance"]["output"]) for ex in batch]
        #     # if self.text_only:
        #     #     model_inputs["labels"] = labels
        #     # else:
        #     #     with self.tokenizer.as_target_tokenizer():
        #     #         labels = self.tokenizer(
        #     #             labels,
        #     #             max_length=self.max_target_length,
        #     #             padding=self.padding,
        #     #             return_tensors=self.return_tensors,
        #     #             truncation=True,
        #     #             pad_to_multiple_of=self.pad_to_multiple_of
        #     #         )
        #     #     label_mask = labels["attention_mask"].bool()
        #     #     model_inputs["labels"] = labels["input_ids"].masked_fill(~label_mask, self.label_pad_token_id)
        # else:
        #     labels = None
            # model_inputs["labels"] = None
        labels = None
        is_causal_lm =  isinstance(self.model, OPTPreTrainedModel) or isinstance(self.model, GPT2PreTrainedModel)
        if is_causal_lm and labels:
            sources = ["".join(sl) for sl in zip(sources, labels)]

        # 2. prepare model inputs first
        if self.text_only:
            model_inputs = {"inputs": sources}
        else:
            model_inputs = self.tokenizer(
                    sources, 
                    max_length=self.max_source_length, 
                    padding=self.padding,
                    return_tensors=self.return_tensors, 
                    truncation=True,
                    pad_to_multiple_of=self.pad_to_multiple_of)
        # 3. prepare model labels second
        if labels and not is_causal_lm:
            if self.text_only:
                model_inputs["labels"] = labels
            else:
                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(
                        labels,
                        max_length=self.max_target_length,
                        padding=self.padding,
                        return_tensors=self.return_tensors,
                        truncation=True,
                        pad_to_multiple_of=self.pad_to_multiple_of
                    )
                label_mask = labels["attention_mask"].bool()
                model_inputs["labels"] = labels["input_ids"].masked_fill(~label_mask, self.label_pad_token_id)
        
        # issue: index & mask doesn't align after deleting tokens(after sentinel tokens)
        # input_ids["denoising objective"] and mask them


        # model_inputs["labels"] = model_inputs["input_ids"].clone()
        # if is_causal_lm:
        #     if self.text_only:
        #         model_inputs["labels"] = model_inputs["inputs"]
        #     else:
        #         model_inputs["labels"] = model_inputs["input_ids"]

        input_ids = model_inputs["input_ids"]
        batch_size, expandend_input_length = input_ids.shape
        # TODO: indexing over batch where bs > 1
        # TODO: indexing over a sequence of token ids for "Definition: " rather 
        # than Definition
        if self.denoise_obj == "definition":
            denoise_obj_token_id = self.tokenizer.encode("Definition:")[0]
        elif self.denoise_obj == "explanation":
            denoise_obj_token_id = self.tokenizer.encode("Explanation:")[0]
        else:
            raise ValueError("Invalid denoise objective: {}".format(self.denoise_obj))
        def_token_indices = (input_ids == denoise_obj_token_id).nonzero()[:, -1]
        if np.count_nonzero(def_token_indices==0) > 0:
            # import pdb; pdb.set_trace()
            # print('')
            raise ValueError("Definition token must not be placed at the beginning of the input")
        noise_input_length = torch.tensor(expandend_input_length).repeat(batch_size) - def_token_indices
        if len(def_token_indices) == 0:
            # since the example is too long and the definition token is not found
            model_inputs["labels"] = torch.tensor([[self.label_pad_token_id]]) # nothing to predict
            return model_inputs
        # determine noise objective length and start index
        try:
            def_mask_indices = [self.random_spans_noise_mask(noise_input_length[i]) for i in range(batch_size)]
        except IndexError:
            import pdb; pdb.set_trace()
            print('')
            raise ValueError("Definition token must not be placed at the beginning of the input")

        
        mask_indices = np.asarray([np.concatenate( (np.asarray([False] * def_token_indices[i].item()), def_mask_indices[i])) for i in range(batch_size)])
        try:
            labels_mask = ~mask_indices
        except TypeError:
            import pdb; pdb.set_trace()
            print('')
            
        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))
        # import pdb; pdb.set_trace()
        # print('check mask results')

        model_inputs["input_ids"] = torch.tensor(self.filter_input_ids(input_ids, input_ids_sentinel), dtype=torch.long)
        model_inputs["labels"] = torch.tensor(self.filter_input_ids(input_ids, labels_sentinel), dtype=torch.long)
        # model_inputs["attention_mask"] = self.filter_input_ids(model_inputs["attention_mask"], input_ids_sentinel)
        model_inputs.pop("attention_mask")

        


        # # prepare decoder_input_ids
        # if self.model is not None and not self.text_only and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
        #     decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])
        #     model_inputs["decoder_input_ids"] = decoder_input_ids
        #     # elif hasattr(self.model, "prepare_inputs_for_generation"): # GPT or OPT model
        #     #     # TODO: delete this
                
        #     #     model_inputs["inputs"] +=\
        #     #          f" Output: {model_inputs.pop('labels')}"
        return model_inputs

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [input_ids, np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32)], axis=-1
        )
        return input_ids

    def random_spans_noise_mask(self, length):

        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.
        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number
        Returns:
            a boolean tensor with shape [length]
        """
        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)


        # two ways, masking the original input and recover based on def token index
        # 
        return is_noise[:orig_length]