from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

text='''
I am Amir Hashmi, a MERN & AI/ML Developer with over 2 years of experience, 
currently working at Devanics.

Detail-oriented Full Stack Developer with strong expertise in Python, Machine 
Learning, Deep Learning, and NLP. Skilled in building full-stack web apps, designing and 
deploying ML models, and integrating AI features into web applications.

Hands-on experience with MERN stack, ASP.Net Core, OOP, DSA, C#, JavaScript, 
and SQL/NoSQL databases. Demonstrated success in professional, internship, freelance, 
and academic projects. Seeking opportunities to contribute to innovative AI/ML solutions 
while leveraging full-stack development expertise.
'''

splitter=RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0
)

chunks= splitter.split_text(text)

print(len(chunks))
print(chunks)