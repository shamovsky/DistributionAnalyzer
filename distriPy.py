import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import base64


class distriPy:
    def __init__(self, distribution_type, params):
        self.distribution_type = distribution_type
        self.params = params

    @st.cache(suppress_st_warning=True)
    def generate_data(self, size=10000):
        data = None
        if self.distribution_type == 'Weibull':
            data = self.generate_weibull(size)
        elif self.distribution_type == 'Geometric':
            data = self.generate_geometric(size)
        elif self.distribution_type == 'Bernoulli':
            data = self.generate_bernoulli(size)
        elif self.distribution_type == 'Binomial':
            data = self.generate_binomial(size)
        elif self.distribution_type == 'Exponential':
            data = self.generate_exponential(size)
        elif self.distribution_type == 'Gamma':
            data = self.generate_gamma(size)
        elif self.distribution_type == 'Normal':
            data = self.generate_normal(size)
        else:
            raise ValueError("Unsupported distribution type.")
        return data

    def generate_weibull(self, size):
        k, lambda_ = self.params
        u = np.random.rand(size)
        data = (1 / lambda_) * (-np.log(1 - u)) ** (1 / k)
        return data

    def generate_geometric(self, size):
        p = self.params
        u = np.random.rand(size)
        data = np.ceil(np.log(u) / np.log(1 - p))
        return data

    def generate_bernoulli(self, size):
        p = self.params
        data = np.random.choice([0, 1], size=size, p=[1 - p, p])
        return data

    def generate_binomial(self, size):
        n, p = self.params
        data = np.random.binomial(n, p, size)
        return data

    def generate_exponential(self, size):
        data = np.random.rand(size)
        return -np.log(1 - data) / self.params

    def generate_gamma(self, size):
        shape, scale = self.params
        size = int(size)
        data = []
        for _ in range(size):
            prod = 1
            for _ in range(int(shape)):
                prod *= np.random.rand()
            data.append(-np.log(prod) * scale)
        return np.array(data)

    def generate_normal(self, size):
        mean, std_dev = self.params
        data = []
        for _ in range(size):
            u1, u2 = np.random.rand(), np.random.rand()
            z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
            data.append(mean + z * std_dev)
        return np.array(data)

    def distribution_facts(self):
        facts = ""
        if self.distribution_type == 'Weibull':
            facts += "Weibull Distribution:\n"
            facts += "  - Used in reliability engineering to model time to failure of devices.\n"
            facts += "  - It's a versatile distribution, capturing a wide range of shapes from exponential to normal.\n"
            facts += "  - It's characterized by a scale parameter (lambda) and a shape parameter (k).\n"
        elif self.distribution_type == 'Geometric':
            facts += "Geometric Distribution:\n"
            facts += "  - Models the number of trials needed until the first success in a sequence of Bernoulli trials.\n"
            facts += "  - It's memoryless, meaning the probability of success in future trials remains the same.\n"
            facts += "  - It has a single parameter, the probability of success in each trial (p).\n"
        elif self.distribution_type == 'Bernoulli':
            facts += "Bernoulli Distribution:\n"
            facts += "  - Represents a random variable with two possible outcomes, usually labeled as success (1) and failure (0).\n"
            facts += "  - It's a special case of the binomial distribution with a single trial.\n"
            facts += "  - It has a single parameter, the probability of success (p).\n"
        elif self.distribution_type == 'Binomial':
            facts += "Binomial Distribution:\n"
            facts += "  - Models the number of successes in a fixed number of independent Bernoulli trials.\n"
            facts += "  - It's characterized by two parameters: the number of trials (n) and the probability of success in each trial (p).\n"
            facts += "  - It's often used in quality control, biology, and finance, among other fields.\n"

        if self.distribution_type == 'Exponential':
            facts += "Exponential Distribution:\n"
            facts += "  - Models the time between events in a Poisson process.\n"
            facts += "  - It's memoryless, meaning the probability of an event occurring in the future is independent of the past.\n"
            facts += "  - It's characterized by a single parameter, the rate (lambda).\n"
        elif self.distribution_type == 'Gamma':
            facts += "Gamma Distribution:\n"
            facts += "  - Generalizes the exponential distribution to multiple shapes.\n"
            facts += "  - It's often used to model the sum of exponentially distributed random variables.\n"
            facts += "  - It has two parameters: shape (k) and scale (theta).\n"
        elif self.distribution_type == 'Normal':
            facts += "Normal Distribution:\n"
            facts += "  - Commonly known as the bell curve or Gaussian distribution.\n"
            facts += "  - It's symmetric and characterized by its mean (mu) and standard deviation (sigma).\n"
            facts += "  - Many natural phenomena approximately follow this distribution.\n"
        else:
            facts += "Unsupported distribution type.\n"
        return facts


# Streamlit app
st.title("Distribution Analyzer")

distribution_type = st.selectbox("Select Distribution",
                                 ['Exponential', 'Gamma', 'Normal', 'Weibull', 'Geometric', 'Bernoulli', 'Binomial'])

if distribution_type:
    params = None
    size = 1000
    st.subheader("Mathematical Formula:")

    if distribution_type == 'Weibull':
        st.latex(
            r'f(x;k,\lambda)=\begin{cases} \frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1}\exp\left(-\left(\frac{x}{\lambda}\right)^{k}\right), & x \geq 0,\\ 0, & x < 0. \end{cases}')

        k = st.slider("Weibull Parameter k", 0.1, 10.0, 2.0, step=0.1)
        lambda_ = st.slider("Weibull Parameter lambda", 0.1, 10.0, 1.0, step=0.1)
        params = (k, lambda_)
    elif distribution_type == 'Exponential':
        st.latex(r'f(x;\lambda)=\lambda e^{-\lambda x}')

        lambda_ = st.slider("Exponential Parameter (lambda)", 0.1, 10.0, 1.0, step=0.1)
        params = lambda_
    elif distribution_type == 'Gamma':
        st.latex(r'f(x;k,\theta)=\frac{1}{\Gamma(k)\theta^k}x^{k-1}e^{-\frac{x}{\theta}}')

        shape = st.slider("Gamma Parameter (k)", 0.1, 10.0, 2.0, step=0.1)
        scale = st.slider("Gamma Parameter (theta)", 0.1, 10.0, 1.0, step=0.1)
        params = (shape, scale)
    elif distribution_type == 'Normal':
        st.latex(r'f(x;\mu,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}')

        mean = st.slider("Normal Parameter (mu)", -10.0, 10.0, 0.0, step=0.1)
        std_dev = st.slider("Normal Parameter (sigma)", 0.1, 10.0, 1.0, step=0.1)
        params = (mean, std_dev)
    elif distribution_type == 'Geometric':
        st.latex(r'f(x;p)=(1-p)^{x-1}p')

        params = st.slider("Geometric Parameter (p)", 0.01, 0.99, 0.3, step=0.01)
    elif distribution_type == 'Bernoulli':
        st.latex(r'f(x;p)=p^x(1-p)^{1-x},\quad x\in \{0, 1\}')

        params = st.slider("Bernoulli Parameter (p)", 0.01, 0.99, 0.5, step=0.01)
    elif distribution_type == 'Binomial':
        st.latex(r'f(x;n,p)=\binom{n}{x}p^x(1-p)^{n-x}')

        n = st.slider("Number of Trials (n)", 1, 100, 10)
        p = st.slider("Binomial Parameter (p)", 0.01, 0.99, 0.3, step=0.01)
        params = (n, p)

    if params:
        analyzer = distriPy(distribution_type, params)

        st.write(analyzer.distribution_facts())

        # Generate data for plotting histogram
        histogram_data = analyzer.generate_data()

        # Plot histogram using Matplotlib
        fig, ax = plt.subplots()
        ax.hist(histogram_data, bins=30, density=True, alpha=0.6, color='g')
        ax.set_title(f'Histogram of {distribution_type} Distribution')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')

        # Define x range
        x = np.linspace(min(histogram_data), max(histogram_data), 100)

        # Add PDF and CDF lines if selected
        add_pdf = st.checkbox("Add PDF to Plot")
        add_cdf = st.checkbox("Add CDF to Plot")


        if add_pdf:
            if distribution_type == 'Weibull':
                y_pdf = (k / lambda_) * (x / lambda_) ** (k - 1) * np.exp(-(x / lambda_) ** k)
                ax.plot(x, y_pdf, label='PDF', color='r')
            elif distribution_type == 'Geometric':
                y_pdf = ((1 - params) ** (x - 1)) * params
                ax.plot(x, y_pdf, label='PMF', color='r')
            elif distribution_type == 'Bernoulli':
                pass
                # y_pdf = params ** x * (1 - params) ** (1 - x)
                # ax.plot([0, 1], y_pdf, label='PMF', color='r')
            elif distribution_type == 'Binomial':
                x_int = np.arange(min(histogram_data), max(histogram_data) + 1)
                y_pdf = np.array([np.math.comb(n, int(i)) * (p ** int(i)) * ((1 - p) ** (n - int(i))) for i in x_int])
                ax.plot(x_int, y_pdf, label='PMF', color='r')

            elif distribution_type == 'Exponential':
                y_pdf = params * np.exp(-params * x)
                ax.plot(x, y_pdf, label='PDF', color='r')
            elif distribution_type == 'Gamma':
                shape, scale = params
                y_pdf = x ** (shape - 1) * np.exp(-x / scale) / (scale ** shape * np.math.gamma(shape))
                ax.plot(x, y_pdf, label='PDF', color='r')
            elif distribution_type == 'Normal':
                mean, std_dev = params
                y_pdf = 1 / (np.sqrt(2 * np.pi * std_dev ** 2)) * np.exp(-(x - mean) ** 2 / (2 * std_dev ** 2))
                ax.plot(x, y_pdf, label='PDF', color='r')
        if add_cdf:
            if distribution_type == 'Weibull':
                y_cdf = 1 - np.exp(-(x / lambda_) ** k)
                ax.plot(x, y_cdf, label='CDF', color='b')
            elif distribution_type == 'Geometric':
                y_cdf = 1 - (1 - params) ** np.floor(x)
                ax.plot(x, y_cdf, label='CDF', color='b')
            elif distribution_type == 'Bernoulli':
                pass
                # y_cdf = np.array([1 - (1 - params) ** i for i in x])
                # ax.plot([0, 1], y_cdf, label='CDF', color='b')
            elif distribution_type == 'Binomial':
                x_int = np.arange(min(histogram_data), max(histogram_data) + 1)  # Convert x to integers
                y_cdf = np.array(
                    [sum(np.array([np.math.comb(n, j) * (p ** j) * ((1 - p) ** (n - j)) for j in range(int(i) + 1)]))
                     for i in x_int])
                ax.plot(x_int, y_cdf, label='CDF', color='b')
            if distribution_type == 'Exponential':
                y_cdf = 1 - np.exp(-params * x)
                ax.plot(x, y_cdf, label='CDF', color='b')
            elif distribution_type == 'Gamma':
                shape, scale = params
                y_cdf = []
                for val in x:
                    integral = 0
                    step = 0.01  # Step size for numerical integration
                    for t in np.arange(0, val, step):
                        integral += (1 / (np.math.gamma(shape) * scale ** shape)) * (t ** (shape - 1)) * np.exp(
                            -t / scale) * step
                    y_cdf.append(integral)

                ax.plot(x, y_cdf, label='CDF', color='b')
            elif distribution_type == 'Normal':
                mean, std_dev = params
                y_cdf = np.array([0.5 * (1 + np.math.erf((i - mean) / (std_dev * np.sqrt(2)))) for i in x])
                ax.plot(x, y_cdf, label='CDF', color='b')

        st.pyplot(fig)
        if st.button("Download Generated Data as CSV"):
                    df = pd.DataFrame(histogram_data, columns=["Generated Data"])
                    csv = df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="{distribution_type}_data.csv">Download CSV File</a>'
                    st.markdown(href, unsafe_allow_html=True)
