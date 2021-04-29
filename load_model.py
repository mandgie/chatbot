from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Save model in model folder
mname = 'facebook/blenderbot-400M-distill'
BlenderbotForConditionalGeneration.from_pretrained(mname).save_pretrained('./model')
BlenderbotTokenizer.from_pretrained(mname).save_pretrained('./model')
