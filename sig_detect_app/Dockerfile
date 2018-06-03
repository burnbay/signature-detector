# docker build -t mockup_app:latest .

FROM ubuntu:18.04
MAINTAINER Lars Erlend Leganger <l.e.leganger@gmail.com>

# Setup framework
RUN apt-get update 
RUN apt-get install -y tesseract-ocr-nor libleptonica-dev imagemagick
RUN apt-get install -y python3-dev python3-pip vim
RUN apt-get install -y nginx supervisor locales

# Set the locale
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8


# Setup flask application
RUN mkdir -p /sig_detect_app
RUN mkdir -p /var/log/sig_detect_app
WORKDIR /sig_detect_app
COPY . /sig_detect_app
RUN pip3 install --no-cache-dir -r requirements.txt

# Setup nginx
RUN rm /etc/nginx/sites-enabled/default
COPY nginx.conf /etc/nginx/nginx.conf
COPY flask.conf /etc/nginx/sites-available/
RUN ln -s /etc/nginx/sites-available/flask.conf /etc/nginx/sites-enabled/flask.conf
RUN echo "daemon off;" >> /etc/nginx/nginx.conf

# Setup supervisord
RUN mkdir -p /var/log/supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Start processes
CMD ["/usr/bin/supervisord"]
