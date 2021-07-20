from datetime import datetime

def doc_ver(ND_doc, upl_doc):
    from PIL import Image
    import imagehash

    im_1= Image.fromarray(ND_doc, 'RGB')
    im_2= Image.fromarray(upl_doc, 'RGB')

    hash = imagehash.average_hash(im_1)
    otherhash = imagehash.average_hash(im_2)
    return (hash - otherhash)



def data_ver(login_details,nadra_details):
#extr_nic_no : extracted nic no (nic no extracted with OCR)
#uploaded_doc: doc image uploaded by customer
#login_details: list of login details like [name, fathername, gender, nicno]
    name = nadra_details[0]
    fname = nadra_details[1]
    dob = nadra_details[2]
    gender = nadra_details[3]
    nic = nadra_details[4]



    if name == login_details[0]:
        return True
        # print("name")
        #if fname == login_details[1]:
            # print("fname")
        #if datetime.strptime(dob, '%Y-%m-%d').date() == datetime.strptime(extr_dob, '%Y-%m-%d').date():
            # print("dob")
            # print(img)
            #if doc_ver(uploaded_doc, nic) < 10:
         #   return True

    return False



