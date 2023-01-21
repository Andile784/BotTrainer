import os

from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

# Define the path to your dataset and validation dataset
'''dataset_path = "C:/Users/andil/projects/TrainBot/dataset"
validation_path = "C:/Users/andil/projects/TrainBot/valiDation"
'''
# Load the pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')


dataset = [
    "At OpendoorsTech, we specialize in solving problems in an efficient and smart way. Our team of experts have a wealth of experience in a range of industries, including Artificial Intelligence. We are dedicated to helping businesses of all sizes succeed in today's digital landscape.",
    "Our services include website design, digital marketing, and branding for businesses of all sizes. We create brand profiles that accurately represent your company and work to connect you with potential customers. Additionally, we offer e-commerce solutions, search engine optimization, and computer support.",
    "We understand the importance of not just increasing your followers but connecting you with your buyers. Our digital marketing strategies are tailored to reach your target audience and help you achieve your business goals. Whether you're looking to increase sales or build brand awareness, we can help.",
    "At OpendoorsTech, we pride ourselves on being leaders in the field of Artificial Intelligence. We have a team of experts who can help you to take advantage of the latest AI technologies and apply them to your business. From chatbots to predictive analytics, we have the knowledge and experience to help you succeed.",
    "We're here for you not only when you have a problem but also when you need advice about your business. Our team is available to answer any questions you may have and provide guidance on how to navigate the digital landscape.",
    "OpendoorsTech is primarily based in South Africa and offers these services through an online presence. Contact us today to see how we can help you and your business succeed. We are confident that our expertise in AI and a wide range of industries will help your business to achieve its goals."
]
validation_data = [
    "At OpendoorsTech, we understand the importance of staying ahead of the curve in today's digital landscape. Our team of experts is constantly researching and experimenting with the latest technologies to bring you cutting-edge solutions for your business.",
    "We specialize in a wide range of industries, including e-commerce, healthcare, and education. Our team has the knowledge and experience to help you succeed in any industry.",
    "Our services include website development, mobile app development, and custom software development. We work closely with our clients to understand their needs and develop solutions that meet those needs.",
    "We also offer digital marketing services, including search engine optimization, social media marketing, and email marketing. We understand the importance of reaching your target audience and driving traffic to your website.",
    "At OpendoorsTech, we pride ourselves on our ability to think outside of the box. We are always looking for new and innovative ways to help our clients succeed. Contact us today to see how we can help you and your business succeed."
]
# Tokenize the text in the dataset and validation dataset
training_data = [{"input_ids": tokenizer.encode(text, add_special_tokens=True, max_length=1024, truncation=True),
                  "labels": tokenizer.encode(text, add_special_tokens=True, max_length=1024, truncation=True)} for text in dataset]
validation_data = [{"input_ids": tokenizer.encode(text, add_special_tokens=True, max_length=1024, truncation=True),
                    "labels": tokenizer.encode(text, add_special_tokens=True, max_length=1024, truncation=True)} for text in validation_data]

# Define the arguments for training
args = {
    "num_train_epochs": 1,
    "evaluate_during_training": True,
    "logging_dir": "logs",
    "save_steps": 100,
    "overwrite_output_dir": True,
    "evaluate_during_training_steps": 100,
    "seed": 42,
    "train_batch_size": 8,
    "eval_batch_size": 8,
}
args = TrainingArguments(**args)

# Convert the dictionary of arguments to an instance of TrainingArguments
training_args = TrainingArguments(
    num_train_epochs=1,
    evaluate_during_training=True,
    logging_dir='logs',
    save_steps=100,
    overwrite_output_dir=True,
    evaluate_during_training_steps=100,
    seed=42,
    train_batch_size=8,
    eval_batch_size=8
)
# Define the trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=training_data,
    eval_dataset=validation_data
)

# Start the fine-tuning process
trainer.train()

# Save the fine-tuned model
model.save_pretrained("gpt2-custom")
