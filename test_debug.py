from src.predict import GrievancePredictor
p = GrievancePredictor()

text_basic = "There is some construction debris on the side of the footpath. It is not blocking traffic but looks messy."
text_loc = "There is some construction debris on the side of the footpath in Delhi. It is not blocking traffic but looks messy."

print(f"Original: {p.predict(text_basic)}")
print(f"With Loc: {p.predict(text_loc)}")
