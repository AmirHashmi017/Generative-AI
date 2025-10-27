from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model1= ChatOpenAI(model="gpt-5")
model2= ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt1= PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2= PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3= PromptTemplate(
    template='Merge the provided notes and quiz ito a single document \n {notes} and {quiz}',
    input_variables=['notes','quiz']
)

parser= StrOutputParser()

parallel_chain= RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 | parser,
}
)

merge_chain = prompt3 | model1 | parser

final_chain= parallel_chain | merge_chain

text='''
Support Vector Machine (SVM) is a powerful supervised machine learning algorithm 
primarily used for classification and regression tasks. Developed by Vladimir Vapnik 
and his colleagues in the 1990s, SVM works by finding the optimal hyperplane that maximally 
separates different classes in a high-dimensional feature space. The algorithm identifies 
support vectors, which are the data points closest to the decision boundary, and uses them 
to define the margin between classes. SVM aims to maximize this margin, making it robust to 
outliers and providing good generalization performance on unseen data. One of the key strengths 
of SVM is the kernel trick, which allows the algorithm to efficiently handle non-linearly 
separable data by implicitly mapping input features into higher-dimensional spaces using 
kernel functions such as linear, polynomial, radial basis function (RBF), and sigmoid. 
SVM is particularly effective in high-dimensional spaces and remains memory efficient 
since it only uses a subset of training points in the decision function. Common applications
of SVM include text classification, image recognition, bioinformatics, and handwriting 
recognition, making it one of the most widely used algorithms in machine learning.
'''

result= final_chain.invoke(text)
print(result)
