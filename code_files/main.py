from infer import generate

print("Your tiny AI is ready. Ask anything from your .txt:")
while True:
    q = input(">> ")
    print("AI:", generate(q))
