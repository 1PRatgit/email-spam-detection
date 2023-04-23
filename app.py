from email.header import Header
from PIL import Image
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
# from streamlit-lottie import st_lottie
import streamlit.components.v1 as components

ps = PorterStemmer()


#Load assets

st.set_page_config(page_title="My Webpage", page_icon=":tada:",layout="wide")

img=Image.open('about.jpg')



# Header

# st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)
st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
<script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
""", unsafe_allow_html=True)



def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email Spam Classifier")

input_sms = st.text_area("Enter the email")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")


# https://source.unsplash.com/LPZy4da9aRo

st.image(img)

# st.markdown("""
# <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.11.6/dist/umd/popper.min.js" integrity="sha384-oBqDVmMz9ATKxIep9tiCxS/Z9fNfEXiDAYTujMAeBAsjFuCZSmKbSSUnQlmh/jp3" crossorigin="anonymous"></script>
# <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.min.js" integrity="sha384-cuYeSxntonz0PPNlHhBs68uyIAVpIIOZZ5JqeqvYYIcEL727kskC66kF92t6Xl2V" crossorigin="anonymous"></script>
# <div id="carouselExampleIndicators" class="carousel slide" data-bs-ride="carousel">
#         <div class="carousel-indicators">
#           <button type="button" data-bs-target="#carouselExampleIndicators" data-bs-slide-to="0" class="active" aria-current="true" aria-label="Slide 1"></button>
#           <button type="button" data-bs-target="#carouselExampleIndicators" data-bs-slide-to="1" aria-label="Slide 2"></button>
#           <button type="button" data-bs-target="#carouselExampleIndicators" data-bs-slide-to="2" aria-label="Slide 3"></button>
#         </div>
#         <div class="carousel-inner">
#           <div class="carousel-item active">
#             <img style=" max-width: 100%;height:670px;"src="https://source.unsplash.com/iIJrUoeRoCQ" class="d-block w-100" alt="...">
#             <div class="carousel-caption">
#               <h5>Identify spam emails</h5>
#                               <p>Spam filters are designed to identify incoming dangerous emails from attackers or marketers.
#                                  Attackers often use emails designed to get you to 
#                                  click on a link that downloads malicious software onto your computer or sends you to a dangerous site.
#                               </p>
#             </div>
#           </div>
#           <div class="carousel-item">
#             <img  style=" max-width: 100%;height:670px;" src="https://source.unsplash.com/LPZy4da9aRo" class="d-block w-100" alt="...">
#             <div class="carousel-caption">
#               <h5>Enhance protection</h5>
#                               <p>Because an email spam filtering can recognize these kinds of emails, it can be a valuable solution for protecting users from unwanted messages. To enhance the protection, some spam filters use insights gained from machine learning to more accurately target junk mail.</p>
                             
#             </div>
#           </div>
#           <div class="carousel-item">
#             <img style="max-width: 100%;height:670px;" src="https://source.unsplash.com/LPZy4da9aRo" class="d-block w-100" alt="...">
#             <div class="carousel-caption">
#               <h5>Extra layer of security</h5>
#               <p>filter out emails that could distract employees or waste their time. Further, because spam folders can be set to automatically delete the emails inside after a certain amount of time.</p>
                              
#             </div>
#           </div>
#         </div>
#         <button class="carousel-control-prev" type="button" data-bs-target="#carouselExampleIndicators" data-bs-slide="prev">
#           <span class="carousel-control-prev-icon" aria-hidden="true"></span>
#           <span class="visually-hidden">Previous</span>
#         </button>
#         <button class="carousel-control-next" type="button" data-bs-target="#carouselExampleIndicators" data-bs-slide="next">
#           <span class="carousel-control-next-icon" aria-hidden="true"></span>
#           <span class="visually-hidden">Next</span>
#         </button>
#       </div>

# """, unsafe_allow_html=True)

st.write("##")
st.write("##")

st.markdown("""
<section id="about" class="about section-padding">
          <div class="container">
              <div class="row">
                  <div class="col-lg-4 col-md-12 col-12">
                      <div class="about-img">
                          <img style="    max-width: 100%;
    height: auto;" src="
                          https://th.bing.com/th/id/OIP.8RILByfFkREQ3yb31022DwHaE8?w=226&h=180&c=7&r=0&o=5&dpr=1.3&pid=1.7
                          " alt="" class="img-fluid">
                      </div>
                  </div>
                  <div class="col-lg-8 col-md-12 col-12 ps-lg-5 mt-md-5">
                      <div class="about-text">
                            <h2>What is it? <br/> Why we need it?</h2>
                            <p>Spam filters detect unsolicited, unwanted, and virus-infested email (called spam) and stop 
                            it from getting into email inboxes.
                             Email is a popular attack vector for hackers and other malicious actors seeking to infect computers with malware. An attacker may send an email with an attachment that looks like an innocent image. However, hidden within the code of the file 
                             could be a virus that is only executable when the recipient clicks on the file’s link. </p>
                            <a href="https://cleantalk.org/spam-stats" class="btn btn-warning">Learn More</a>
                      </div>
                  </div>
              </div>
          </div>
      </section>
""",unsafe_allow_html=True)

st.write("##")

st.markdown("""
<div class="row">
    <div class="col-md-12">
        <div class="section-header text-center pb-5">
            <h2>Statistical analysis</h2>
            <p>Results obtained is represented in graphical form below by analysing output.</p>
        </div>
    </div>
  </div>
""",unsafe_allow_html=True)


st.write("##")

st.markdown("""

<div class="container text-center">
  <div class="row">
    <div class="col">
      <div class="section-header text-center pb-5">
           <div class="img-area mb-4">
              <img src="https://media.geeksforgeeks.org/wp-content/uploads/20220912123608/img1.png" class="img-fluid" alt="">
          </div>
          <p class="lead">From the above graph, we can conclude that the model is built accurately</p>
    </div> 
    </div>
    <div class="col">
      <div class="section-header text-center pb-5">
           <div class="img-area mb-4">
              <img src="https://th.bing.com/th/id/OIP.KV6_0XulF-6MMOq3vtDXdgAAAA?w=260&h=187&c=7&r=0&o=5&dpr=1.3&pid=1.7" class="img-fluid" alt="">
          </div>
          <p class="lead">The plot above shows, the accuracy is increasing over increasing epochs. As expected, the model is performing better in the training set than the validation set.</p>
    </div> 
    </div>
    <div class="col">
      <div class="section-header text-center pb-5">
           <div class="img-area mb-4">
              <img src="https://th.bing.com/th/id/OIP.lMnCCGdv8-15uPhxP9ZPeAHaEk?pid=ImgDet&rs=1" class="img-fluid" alt="">
          </div>
          <p class="lead">By plotting a loss vs. accuracy from the model we can see our loss going down per epoch while accuracy is increased with Particle swarm optimisation.</p>
    </div> 
    </div>
  </div>
</div>
""",unsafe_allow_html=True)
