# 1. Type Dict don't enforce Data Validation but Pydantic enforces it
# 2. It can do type coercing means if we pass 32 in string and age is int it will try to convert to int means coerce type and don't throw error
# 3. It can apply some built in validations like email validation
# 4. It can add constraints


from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = 'ashir'
    age: Optional[int]= None
    email: EmailStr
    cgpa: float =Field(gt=0,lt=4.1, default=3.2, description="Decimal value representing cgpa of student")

new_student = {'name':"amir",'email':'abc@gmail.com','cgpa':3.8}

student= Student(**new_student)

print(student)
print(type(student))

# Can convert pydantic to dictionary
student_dict= dict(student)
print(student_dict['name'])

# Can convert Pydantic to json
student_json= student.model_dump_json()
print(student_json)