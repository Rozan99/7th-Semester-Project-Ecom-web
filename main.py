import uuid
from flask import Flask, request, render_template, session, url_for, redirect, jsonify
import random
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
import pandas as pd
import math
import hmac
import hashlib
import base64

app = Flask(__name__)

#for database mysqlclient and flasksqlalchemy is installed

#database configuration
app.secret_key = "00000"
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:@localhost:3306/ecom?ssl_disabled=true"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class DisplayProduct(db.Model):
    __tablename__ = 'displayproduct'

    pid = db.Column(db.Integer, primary_key=True, autoincrement=True)
    pname = db.Column(db.String(255), nullable=False)
    reviewcount = db.Column(db.Float, nullable=False)
    brand = db.Column(db.String(255), nullable=False)
    imageurl = db.Column(db.Text, nullable=False)
    rating = db.Column(db.Float, nullable=False)
    description=db.Column(db.Text,nullable=False)
    price = db.Column(db.Integer, nullable=False) 
    def __repr__(self):
        return f'<DisplayProduct {self.pname}>'


class Products(db.Model):
    __tablename__ = 'products'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    productid = db.Column(db.String(255), nullable=False)  # Primary key with auto increment
    productname = db.Column(db.String(255), nullable=False)  # Product name (varchar)
    reviewcount = db.Column(db.Float, nullable=False)  # Review count (float)
    productbrand = db.Column(db.String(255), nullable=False)  # Brand name (varchar)
    imageurl = db.Column(db.Text, nullable=False)  # Image URL (text)
    rating = db.Column(db.Float, nullable=False)  # Rating (float)
    description = db.Column(db.Text, nullable=False)
    price = db.Column(db.Integer, nullable=False)  # Price (integer)

    def __repr__(self):
        return f'<Products {self.pname}>'


class Cart(db.Model):
    __tablename__ = 'carts'
    cartid = db.Column(db.Integer, primary_key=True, autoincrement=True)  # Primary key with auto increment
    userid = db.Column(db.Integer, db.ForeignKey('signup.id'), nullable=False)  # Primary key with auto increment
    productid = db.Column(db.String(255), nullable=False)  # Product name (varchar)
    productname = db.Column(db.String(255), nullable=False)  # Product name (varchar)
    quantity = db.Column(db.Integer, nullable=False)  # Review count (float)
    image = db.Column(db.String(255), nullable=False)  # Image URL (text)
    price = db.Column(db.Integer, nullable=False)  # Price (integer)
    user = db.relationship('Signup', back_populates='carts')

    def __repr__(self):
        return f'<Products {self.pname}>'



class Signup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)
    repassword= db.Column(db.String(100),nullable=False)
    status=db.Column(db.Integer,nullable=False)
    role= db.Column(db.String(255),nullable=False)
    carts = db.relationship('Cart', back_populates='user')



class Signin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)
    status=db.Column(db.Integer,nullable=False)

class Admin(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        adminName = db.Column(db.String(255), nullable=False)
        adminPassword = db.Column(db.String(255), nullable=False)
        role = db.Column(db.String(255), nullable=False)

trending_products = pd.read_csv("models/trending_products.csv")
train_data = pd.read_csv("models/clean_dataset.csv")



def gen_sha256(key, message):
    key = key.encode('utf-8')
    message = message.encode('utf-8')
    hmac_sha256 = hmac.new(key, message, hashlib.sha256)
    signature = base64.b64encode(hmac_sha256.digest()).decode('utf-8')
    return signature


def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text


def compute_tf_idf(train_data):
    term_freqs = []
    doc_freqs = {}
    num_documents = len(train_data)

    for tags in train_data['Tags']:
        terms = tags.lower().split()
        term_count = {}
        for term in terms:
            term_count[term] = term_count.get(term, 0) + 1
        term_freqs.append(term_count)
        for term in term_count.keys():
            doc_freqs[term] = doc_freqs.get(term, 0) + 1

    tf_idf_matrix = []
    for term_count in term_freqs:
        tf_idf_vector = {}
        for term, count in term_count.items():
            tf = count / len(term_count)
            idf = math.log(num_documents / (1 + doc_freqs[term]))
            tf_idf_vector[term] = tf * idf
        tf_idf_matrix.append(tf_idf_vector)
    return tf_idf_matrix


def cosine_similarity(vector1, vector2):

    dot_product = sum(vector1.get(term, 0) * vector2.get(term, 0) for term in vector1)
    norm1 = math.sqrt(sum(v ** 2 for v in vector1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in vector2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def content_based_recommendations(train_data, item_name, top_n=10):
    if item_name not in train_data['Name'].values:
        return pd.DataFrame()

    tf_idf_matrix = compute_tf_idf(train_data)
    item_index = train_data[train_data['Name'] == item_name].index[0]
    similarities = []
    for i, tf_idf_vector in enumerate(tf_idf_matrix):
        similarity = cosine_similarity(tf_idf_matrix[item_index], tf_idf_vector)
        similarities.append((i, similarity))
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    print(similarities)
    top_similar_indices = [idx for idx, _ in similarities[1:top_n + 1]]
    print(top_similar_indices)
    recommended_items_details = train_data.iloc[top_similar_indices][
        ['ID','Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating','Price','Description']]
    return recommended_items_details



#create route
@app.route("/")
def index():
    products = DisplayProduct.query.all()
    return render_template('index.html', data=products)


@app.route("/main")
def main():
    return render_template('main.html',content_based_rec = None, truncate= truncate)


@app.route("/index")
def indexredirect():
    products = DisplayProduct.query.all()
    return render_template('index.html',
                           data=products
                          )


@app.route("/fetch")
def fetch():
    products = DisplayProduct.query.all()
    return render_template('fetch.html', data=products)


@app.route('/product/<string:pid>')
def product_detail(pid):
    query = text("SELECT * FROM displayproduct WHERE pid = :pid")
    result = db.session.execute(query, {'pid': pid}).fetchone()

    if result:
        pname = result[1]
        product_info = {
            'ID': result[0],
            'Name': result[1],
            'ReviewCount': result[2],
            'Brand': result[3],
            'ImageURL': result[4],
            'Rating': result[5],
            'Description': result[6] if len(result) > 7 else "No description available.",
            'Price': result[8] if len(result) > 6 else None,  # Adjust if Price column exists
        }
    else:
        result_db = train_data[train_data['ID'] == pid]
        if not result_db.empty:
            result = result_db.iloc[0]
            pname = result['Name']
            product_info = {
                'ID': result['ID'],
                'Name': result['Name'],
                'ReviewCount': result['ReviewCount'],
                'Brand': result['Brand'],
                'ImageURL': result['ImageURL'],
                'Rating': result['Rating'],
                'Description': result['Description'],  # Default description
                'Price': result['Price'] if 'Price' in result else None
            }
        else:
            return "Product not found", 404

    recommendations1 = content_based_recommendations(train_data, pname, top_n=5)


    if pname not in train_data['Name'].values:
        message = f"No recommendations available for the product '{pname}' as it is not found in the dataset."
    else:
        message = None

    return render_template(
        'product_detail.html',
        product=product_info,
        recommendations1=recommendations1.to_dict(orient='records'),
        message=message
    )



# for signup route
@app.route("/signup", methods=['POST','GET'])
def signup():
    products = DisplayProduct.query.all()  # For rendering products

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        repassword = request.form.get('repassword', '').strip()

        # --- Validations ---
        if not username or not email or not password or not repassword:
            return render_template('index.html', data=products, signup_error="All fields are required")

      
        import re
        username_pattern = r'^[A-Za-z][A-Za-z0-9_]{2,}$'  

        if not re.match(username_pattern, username):
            return render_template('index.html', data=products, signup_error="Username must start with a letter and be at least 3 characters long")

        # Check if username exists
        user = Signup.query.filter_by(username=username).first()
        if user:
            return render_template('index.html', data=products, signup_error="Username already exists, try again")

        # Check if email exists
        email_check = Signup.query.filter_by(email=email).first()
        if email_check:
            return render_template('index.html', data=products, signup_error="Email already registered, try again")

        # Check valid email format
        email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        if not re.match(email_pattern, email):
            return render_template('index.html', data=products, signup_error="Invalid email address")

        # Password length
        if len(password) < 8:
            return render_template('index.html', data=products, signup_error="Password must be at least 8 characters long")

        # Password match
        if password != repassword:
            return render_template('index.html', data=products, signup_error="Passwords do not match")

        # --- Save user ---
        new_signup = Signup(
            username=username,
            email=email,
            password=password,
            repassword=repassword,
            status=1,
            role='user'
        )
        db.session.add(new_signup)
        db.session.commit()

        return render_template('index.html', data=products, signup_success="User signed up successfully")

    return render_template('index.html', data=products)



# for signin

@app.route("/signin", methods=['GET','POST'])
def signin():
    products = DisplayProduct.query.all()

    if request.method == 'POST':
        username = request.form.get('signinUsername', '').strip()
        password = request.form.get('signinPassword', '').strip()

        if not username or not password:
            return render_template('index.html', data=products, signin_error="All fields are required")

        user = Signup.query.filter_by(username=username, status=1, role="user").first()
        if user:
            # For testing with plain text
            if user.password == password:
                session['userid'] = user.id
                session['username'] = user.username
                session['logged_in'] = True
                return render_template('index.html', data=products, signin_success=f"Welcome {user.username}")
            else:
                return render_template('index.html', data=products, signin_error="Invalid Password")

        # Admin login
        admin = Admin.query.filter_by(adminName=username, role="admin").first()
        if admin:
            if admin.adminPassword == password:
                session['adminlogin'] = admin.id
                session['admin'] = True
                return render_template('./admin/admin.html', signin_success="Welcome Admin")
            else:
                return render_template('index.html', data=products, signin_error="Invalid Admin Password")

        return render_template('index.html', data=products, signin_error="Invalid Username")

    # GET request
    return render_template('index.html', data=products)


@app.route("/logout")
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('index'))

@app.route("/suggestions", methods=["GET"])
def suggestions():
    query = request.args.get("query", "").strip()
    if not query:
        return jsonify({"suggestions": []})
    results = Products.query.filter(
        Products.productname.ilike(f"%{query}%")
    ).limit(10).all()
    return jsonify({"suggestions": [product.productname for product in results]})

#comment

@app.route("/search", methods=['GET'])
def search():
    query = request.args.get('query')
    if query:

        results = Products.query.filter(Products.productname.ilike(f"%{query}%")).all()
        recommendations1 = content_based_recommendations(train_data, query, top_n=10)

        message=None
        if recommendations1.empty:
            message=f"No recommendations found for '{query}'. Item not in the training data."

        return render_template('search.html', results=results,  recommendations1=recommendations1.to_dict(orient='records'),message=message)
    else:
        return "No search query provided.", 400


@app.route("/cart", methods=['POST', 'GET'])
def cart():
    if 'logged_in' not in session:
        return redirect(url_for('signin'))
    else:
        user= session['userid']

    pid = request.form.get("pid")
    pname = request.form.get("pname")
    price = request.form.get("price")
    image = request.form.get("image")

    if pid and pname and price and image:
        check = text("Select * from carts where userid = :userid and productid = :productid ")
        checkresult = db.session.execute(check,{'userid':user,'productid':pid}).fetchone()

        if checkresult:
            updata = text("Update carts set quantity = quantity+1 where userid = :userid and productid = :productid")
            db.session.execute(updata,{'userid':user,'productid':pid})
            db.session.commit()

        else:
            adddata = text("""
                INSERT INTO carts (userid, productid, productname, quantity, image, price) 
                VALUES (:userid, :productid, :productname, 1, :image, :price)
            """)
            db.session.execute(adddata,{'userid':user,'productid':pid,'productname':pname,'image':image,'price':price})
            db.session.commit()

    query = text("Select * from carts where userid =  :userid")
    result = db.session.execute(query, {'userid': user}).fetchall()

    cart_items = []
    if result:
        for item in result:
            cart_items.append({
                'pid': item[2],
                'pname': item[3],
                'quantity': item[4],
                'image': item[5],
                'price': item[6]
            })

    return render_template('cart.html', cartdata= cart_items)

@app.route('/checkout', methods=['POST', 'GET'])
def checkout():
    pid = request.values.get('pid')
    pname = request.values.get('pname')
    price = float(request.values.get('price', 0))
    quantity = int(request.values.get('quantity', 1))
    image = request.values.get('image')
    user = session.get('userid')

    if not all([pid, pname, price, quantity, user]):
        flash("Invalid checkout data", "danger")
        return redirect(url_for("cart"))

    totalprice = price * quantity

    # Move to purchase table
    insertquery = text("""
        INSERT INTO purchase (productid, productname, quantity, productprice, userid)
        VALUES (:productid, :productname, :quantity, :productprice, :userid)
    """)
    db.session.execute(insertquery, {
        'productid': pid,
        'productname': pname,
        'quantity': quantity,
        'productprice': totalprice,
        'userid': user
    })
    db.session.commit()

    # Remove from cart
    delquery = text("DELETE FROM carts WHERE userid=:userid AND productid=:productid")
    db.session.execute(delquery, {'userid': user, 'productid': pid})
    db.session.commit()

    # Prepare eSewa info
    esewa_info = {
        'amount': str(totalprice),
        'tax_amount': '0',
        'total_amount': str(totalprice),
        'transaction_uuid': f"{user}-{pid}-{uuid.uuid4()}",
        'product_code': 'EPAYTEST',
        'product_service_charge': '0',
        'product_delivery_charge': '0',
        'success_url': url_for('esewa_success', _external=True),
        'failure_url': url_for('esewa_failure', _external=True),
        'signed_field_names': 'total_amount,transaction_uuid,product_code',
        'pid': pid,
        'pname': pname,
        'price': str(totalprice),
        'quantity': str(quantity),
        'image': image
    }

    secret_key = "8gBm/:&EnhH.1/q"
    data_to_sign = f"total_amount={esewa_info['total_amount']},transaction_uuid={esewa_info['transaction_uuid']},product_code={esewa_info['product_code']}"
    esewa_info['signature'] = gen_sha256(secret_key, data_to_sign)

    session['esewa_info'] = esewa_info

    return render_template('checkout.html', esewa_info=esewa_info)



@app.route('/esewa_success', methods=['GET', 'POST'])
def esewa_success():
    esewa_info = session.pop('esewa_info', None)
   

    return render_template('success.html',message="Payment Successful. Thank you for your purchase!",esewa_info=esewa_info)

@app.route('/esewa_failure', methods=['GET', 'POST'])
def esewa_failure():
    if 'logged_in' not in session:
        return redirect(url_for('signin'))
    else:
        user = session['userid']
    pid = request.form.get("pid")
    pname = request.form.get("pname")
    price = request.form.get("price")
    image = request.form.get("image")

    if pid and pname and price and image:
        check = text("Select * from carts where userid = :userid and productid = :productid ")
        checkresult = db.session.execute(check, {'userid': user, 'productid': pid}).fetchone()

        if checkresult:
            updata = text("Update carts set quantity = quantity+1 where userid = :userid and productid = :productid")
            db.session.execute(updata, {'userid': user, 'productid': pid})
            db.session.commit()

        else:
            adddata = text("""
              INSERT INTO carts (userid, productid, productname, quantity, image, price) 
              VALUES (:userid, :productid, :productname, 1, :image, :price)
          """)
            db.session.execute(adddata,
                               {'userid': user, 'productid': pid, 'productname': pname, 'image': image, 'price': price})
            db.session.commit()

    query = text("Select * from carts where userid =  :userid")
    result = db.session.execute(query, {'userid': user}).fetchall()

    cart_items = []
    if result:
        for item in result:
            cart_items.append({
                'pid': item[2],
                'pname': item[3],
                'quantity': item[4],
                'image': item[5],
                'price': item[6]
            })

    return render_template('cart.html', cartdata=cart_items)

@app.route('/removeItem',methods=['POST'])
def removeitem():
    pid = request.form.get("pid")
    user = session['userid']
    query= text("delete from carts where userid =:userid and productid=:productid ")
    db.session.execute(query,{'userid':user,'productid':pid})
    db.session.commit()


    return redirect(url_for('cart'))

@app.route('/detail')
def detail():
    if 'logged_in' not in session:
        return render_template('signin.html')

    user=session['userid']

    query= text("select * from purchase where userid=:userid")
    result=db.session.execute(query,{'userid':user}).fetchall()
    db.session.commit()

    return render_template('detail.html',result=result)


@app.route("/admin")
def admin():
    selectcount = text("select count(purchaseid) from purchase ")
    count = db.session.execute(selectcount).fetchone()

    newpurchase= text("select * from purchase order by purchaseid DESC limit 2")
    purchasecount = db.session.execute(newpurchase).fetchall()

    newuser = text("select * from signup order by id DESC limit 2")
    usercount = db.session.execute(newuser).fetchall()

    selectproduct = text("select count(ID) from products ")
    prodcount = db.session.execute(selectproduct).scalar()


    selectproduct1 = text("select count(pid) from displayproduct ")
    prodcount1 = db.session.execute(selectproduct1).scalar()


    value=prodcount1+prodcount

    price= text("select sum(productprice) from purchase")
    pricecount = db.session.execute(price).scalar()


    selectuser = text("select count(id) from signup ")
    user = db.session.execute(selectuser).fetchone()

    return render_template('./admin/admin.html',totalcount=count,user=user,product=value,price= pricecount,newuser=usercount,newpurchase=purchasecount)


@app.route("/adminusers")
def adminusers():

    query = text("Select * from signup where status =1")
    user= db.session.execute(query).fetchall()
    db.session.commit()

    return render_template('./admin/adminusers.html',users=user)


@app.route("/adminlogout")
def adminlogout():
    session.pop('adminlogin',None)

    return  redirect(url_for('index'))

@app.route("/adminusersdeactive")
def adminusersdeactive():
    query = text("Select * from signup where status =0")
    user = db.session.execute(query).fetchall()
    db.session.commit()
    return render_template('./admin/adminusersdeactive.html',users=user)


@app.route("/activateuser",methods=['Post'])
def activateuser():
    userid = request.form.get('userid')
    username = request.form.get('username')
    query = text("update signup set status = 1 where id=:userid and username=:username")
    db.session.execute(query, {'userid': userid, 'username': username})
    db.session.commit()

    return render_template('admin/activateuser.html')


@app.route("/removeuser",methods=['POST','GET'])
def removeuser():
    userid = request.form.get('userid')
    username=request.form.get('username')
    # query= text("Delete from signup where id=:userid and username=:username")
    query= text("update signup set status = 0 where id=:userid and username=:username")
    db.session.execute(query,{'userid':userid,'username':username})
    db.session.commit()

    return render_template('admin/removeuser.html')


@app.route("/products")
def products():

    query = text("Select * from displayproduct")
    result = db.session.execute(query).fetchall()
    db.session.commit()

    query2= text("Select * from products")
    result2= db.session.execute(query2).fetchall()
    db.session.commit()

    return render_template('admin/products.html',result=result,result2=result2)

@app.route("/editproduct", methods=['POST'])
def editproduct():
    productid = request.form.get("productid")
    productname = request.form.get("productname")

    # Fetch from displayproduct
    query = text("SELECT * FROM displayproduct WHERE pid=:productid AND pname=:productname")
    result = db.session.execute(query, {'productid': productid, 'productname': productname}).fetchall()

    # Fetch from products
    query1 = text("SELECT * FROM products WHERE productId=:pid AND productname=:pname")
    result2 = db.session.execute(query1, {'pid': productid, 'pname': productname}).fetchall()

    return render_template('admin/editproduct.html', result=result, datas=result2)

@app.route("/changedata",methods=['Post'])
def changedata():

    pid= request.form.get('id')
    pname=request.form.get('pname')
    reviewcount=request.form.get('reviewcount')
    brand=request.form.get('brand')
    imageurl=request.form.get('imageurl')
    rating=request.form.get('rating')
    description=request.form.get('description')
    category=request.form.get('category')
    price=request.form.get('price')

    query= text("Select * from displayproduct where pid=:pid and pname=:pname")
    result = db.session.execute(query,{'pid':pid,'pname':pname})
    db.session.commit()

    query1 = text("Select * from products where productId=:pid and productname=:pname")
    result1 = db.session.execute(query1, {'pid': pid, 'pname': pname})
    db.session.commit()

    if result:

        upquery = text(
            "Update displayproduct set pname=:pname, reviewcount=:reviewcount, brand=:brand, imageurl=:imageurl,"
            "rating=:rating, description=:description, category=:category, price=:price where pid=:pid")
        db.session.execute(upquery,
                                    {'pid': pid,
                                    'pname': pname,
                                     'reviewcount': reviewcount,
                                     'brand': brand,
                                     'imageurl': imageurl,
                                    'rating': rating,
                                     'description': description,
                                     'category': category,
                                    'price': price }
                           )
        db.session.commit()


    if result1:
        upquery2 = text(
            "Update products set productname=:pname, reviewcount=:reviewcount, productbrand=:brand, imageurl=:imageurl,"
            "rating=:rating, description=:description, category=:category, price=:price where productId=:productId ")
        db.session.execute(upquery2,
                                       {'productId': pid,'pname': pname, 'reviewcount': reviewcount, 'brand': brand,
                                        'imageurl': imageurl,
                                        'rating': rating, 'description': description, 'category': category,
                                        'price': price })
        db.session.commit()
    return render_template('admin/changedata.html')
@app.route("/editproduct", methods=["GET"])
def edit_product():
    # Example: load all products to edit
    products = db.session.execute(text("SELECT * FROM displayproduct")).fetchall()
    return render_template("admin/editproduct.html", products=products)



# deleting the product---------------------------------------------------------------
@app.route("/delete", methods=["POST"])
def delete():
    product_input = request.form.get('productid', '').strip()

    if not product_input:
        return "Missing product ID or Name", 400

    product = None
    table = None
    pid = None

    # Try numeric ID first
    if product_input.isdigit():
        pid = int(product_input)
        product = db.session.execute(
            text("SELECT * FROM products WHERE productId=:pid"),
            {'pid': pid}
        ).fetchone()
        table = "products"

    # If not found, try displayproduct by name
    if not product:
        product = db.session.execute(
            text("SELECT * FROM displayproduct WHERE LOWER(pname)=LOWER(:pname)"),
            {'pname': product_input}
        ).fetchone()
        table = "displayproduct" if product else None

    if product:
        try:
            if table == "products":
                db.session.execute(
                    text("DELETE FROM products WHERE productId=:pid"),
                    {'pid': pid}
                )
            else:
                db.session.execute(
                    text("DELETE FROM displayproduct WHERE LOWER(pname)=LOWER(:pname)"),
                    {'pname': product_input}
                )
            db.session.commit()
            print(f"Deleted {product_input} from {table}")
            return redirect(url_for("delete_result", pid=pid or "", pname=product_input if table=="displayproduct" else "", table=table))
        except Exception as e:
            db.session.rollback()
            print("Deletion failed:", e)
            return f"Error deleting product: {str(e)}", 500
    else:
        print(f"Product not found: {product_input}")
        return redirect(url_for("delete_result", pid="", pname=product_input, table="notfound"))

    

# ---------------- Delete Result Page -----------------
@app.route("/delete-result")
def delete_result():
    pid = request.args.get("pid")
    pname = request.args.get("pname")
    table = request.args.get("table")

    # Fetch updated products list
    products = db.session.execute(text("SELECT * FROM products")).fetchall()
    displayproducts = db.session.execute(text("SELECT * FROM displayproduct")).fetchall()

    return render_template(
        "admin/deleteproduct.html",
        pid=pid,
        pname=pname,
        table=table,
        products=products,
        displayproducts=displayproducts
    )


# Purchase

@app.route("/purchase")
def purchase():
    query = text("Select * from purchase")
    result = db.session.execute(query).fetchall()

    namelist=[]
    for name in result:
        userid=name.userid
        namelist.append(userid)


    if namelist:
        namequery = text("Select id, username from signup where id in :id")
        nameresult=db.session.execute(namequery,{'id':tuple(namelist)}).fetchall()
        db.session.commit()

    return render_template('admin/purchase.html',results=result,names=nameresult)

@app.route("/addproduct")
def addproduct():

    return render_template('admin/addproduct.html')

@app.route("/insert", methods=['POST'])
def insert():
    # Get form values and strip whitespace
    id = request.form.get('id', '').strip()
    name = request.form.get('name', '').strip()
    reviewcount = request.form.get('reviewcount', '').strip()
    brand = request.form.get('brand', '').strip()
    imageurl = request.form.get('imageurl', '').strip()
    rating = request.form.get('rating', '').strip()
    description = request.form.get('description', '').strip()
    category = request.form.get('category', '').strip()
    price = request.form.get('price', '').strip()

    # Check for empty fields
    if not id or not name or not reviewcount or not brand or not imageurl or not rating or not description or not category or not price:
        message = "All fields are required"
        return render_template('admin/addproduct.html', message=message)

    try:
        # Numeric ID -> insert into products table
        if id.isdigit():
            query = text("""
                INSERT INTO products 
                (productId, productname, reviewcount, productbrand, imageurl, rating, description, category, price)
                VALUES (:id, :name, :reviewcount, :brand, :imageurl, :rating, :description, :category, :price)
            """)
            db.session.execute(query, {
                'id': id, 'name': name, 'reviewcount': reviewcount, 'brand': brand,
                'imageurl': imageurl, 'rating': rating, 'description': description,
                'category': category, 'price': price
            })

        # Non-numeric ID -> insert into displayproduct table
        else:
            query = text("""
                INSERT INTO displayproduct 
                (pname, reviewcount, brand, imageurl, rating, description, category, price)
                VALUES (:name, :reviewcount, :brand, :imageurl, :rating, :description, :category, :price)
            """)
            db.session.execute(query, {
                'name': name, 'reviewcount': reviewcount, 'brand': brand,
                'imageurl': imageurl, 'rating': rating, 'description': description,
                'category': category, 'price': price
            })

        db.session.commit()
        message = "Product added successfully"
        return render_template('admin/addproduct.html', message=message)

    except Exception as e:
        db.session.rollback()
        message = f"Error inserting product: {str(e)}"
        return render_template('admin/addproduct.html', message=message)

@app.route("/category")
def category():

    query=text("Select * from category")
    result= db.session.execute(query).fetchall()



    return render_template('admin/category.html', result=result)



@app.route("/deletecategory",methods=['post'])
def deletecategory():
    pname= request.form.get('productname')
    pid= request.form.get('productid')

    query= text("Delete from category where id=:pid and Categories=:pname")
    result=db.session.execute(query,{'pid':pid,'pname':pname})
    db.session.commit()
    if result:
        print("Eyaihhh")

    return render_template('admin/deletecategory.html')

@app.route("/addcategory")
def addcategory():
    return render_template('admin/addcategory.html')


@app.route("/insertcategory", methods=['POST'])
def insertcategory():
    cname = request.form.get('name')

    # Check if value is empty
    if not cname or cname.strip() == "":
        return render_template(
            'admin/addcategory.html',  
            message="Category name cannot be empty!"
        )

    try:
        # Insert into DB
        query = text("INSERT INTO category (Categories) VALUES (:name)")
        db.session.execute(query, {'name': cname.strip()})
        db.session.commit()

        # Success message
        return render_template(
            'admin/addcategory.html', 
            message=f"Category '{cname}' added successfully!"
        )

    except Exception as e:
        db.session.rollback()
        return render_template(
            'admin/addcategory.html',  
            message=f"Error inserting category: {str(e)}"
        )

@app.route("/heads")
def heads():
    return render_template('heads.html')


@app.route("/foots")
def foots():
    return render_template('foots.html')

@app.route("/categories")
def categories():

    return render_template('categories.html')


@app.route("/showinfo",methods=['Post'])
def showinfo():

    category=request.form.get('category')


    if category=='all':
        query=text("select * from products ")
        result=db.session.execute(query)
    else:
        query = text("SELECT * FROM products WHERE category LIKE :category;")
        result = db.session.execute(query, {'category': f'%{category}%'})

    return render_template('categories.html',products1=result)

@app.route("/categorybutton")
def categorybutton():

    return render_template('categorybutton.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/services")
def services():
    return render_template('services.html')

#duita folder hunai parxa static ma images, video, css hunuparxa
#templates ma html

if __name__ == '__main__':
    app.run(debug=True)
