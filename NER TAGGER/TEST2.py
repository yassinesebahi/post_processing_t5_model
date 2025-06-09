text = "einschließlich der anzurechnenden Untersuchungshaft, vgl. BGH NStZ-RR 2008, 182"

# NER tagging
tagged = tag_sentence(text)
for item in tagged:
    print(item)

# Citation classification
classified = classify_sentence(text)
for item in classified:
    print(item)