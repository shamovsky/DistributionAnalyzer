import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min, geom, bernoulli, binom, expon, norm, gamma
import streamlit as st

class DistributionAnalyzer:
    def __init__(self, distribution_type, params):
        self.distribution_type = distribution_type
        self.params = params

    @st.cache(suppress_st_warning=True)
    def generate_data(self, size=1000):
        data = None
        if self.distribution_type == 'Weibull':
            data = weibull_min.rvs(*self.params, size=size)
        elif self.distribution_type == 'Geometric':
            params = (self.params,)  # Wrap the single parameter in a tuple
            data = geom.rvs(*params, size=size)
        elif self.distribution_type == 'Bernoulli':
            params = (self.params,)
            data = bernoulli.rvs(params, size=size)
        elif self.distribution_type == 'Binomial':
            n, p = self.params
            data = binom.rvs(n, p, size=size)
        elif self.distribution_type == 'Exponential':
            data = expon.rvs(scale=self.params, size=size)
        elif self.distribution_type == 'Normal':
            data = norm.rvs(loc=self.params[0], scale=self.params[1], size=size)
        elif self.distribution_type == 'Gamma':
            shape, scale = self.params
            data = gamma.rvs(shape, scale=scale, size=size)
        else:
            raise ValueError("Unsupported distribution type.")
        return data

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
        elif self.distribution_type == 'Exponential':
            facts += "Exponential Distribution:\n"
            facts += "  - Models the time between events in a Poisson process, such as the arrival of customers in a queue.\n"
            facts += "  - It's characterized by a single parameter, the rate of occurrence (lambda).\n"
        elif self.distribution_type == 'Normal':
            facts += "Normal Distribution:\n"
            facts += "  - Often referred to as the bell curve or Gaussian distribution.\n"
            facts += "  - It's characterized by two parameters: the mean (mu) and the standard deviation (sigma).\n"
            facts += "  - It's widely used in natural and social sciences to represent real-valued random variables.\n"
        elif self.distribution_type == 'Gamma':
            facts += "Gamma Distribution:\n"
            facts += "  - Generalization of the exponential distribution, allowing for a more flexible shape.\n"
            facts += "  - It's characterized by two parameters: the shape (k) and the scale (theta or 1/lambda).\n"
            facts += "  - It's used in queuing theory, reliability, and finance, among other areas.\n"
        else:
            facts += "Unsupported distribution type.\n"
        return facts


# Streamlit app
st.title("DistriPy")

distribution_type = st.selectbox("Select Distribution", ['Weibull', 'Geometric', 'Bernoulli', 'Binomial', 'Exponential', 'Normal', 'Gamma'])

if distribution_type:
    params = None
    size = 1000
    st.subheader("Mathematical Formula:")

    if distribution_type == 'Weibull':
        st.latex(r'f(x;k,\lambda)=\begin{cases} \frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1}\exp\left(-\left(\frac{x}{\lambda}\right)^{k}\right), & x \geq 0,\\ 0, & x < 0. \end{cases}')

        k = st.slider("Weibull Parameter k", 0.1, 10.0, 2.0, step=0.1)
        lambda_ = st.slider("Weibull Parameter lambda", 0.1, 10.0, 1.0, step=0.1)
        params = (k, lambda_)
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
    elif distribution_type == 'Exponential':
        st.latex(r'f(x;\lambda) = \lambda e^{-\lambda x}, \quad x \geq 0')

        lambd = st.slider("Exponential Parameter (lambda)", 0.01, 10.0, 1.0, step=0.01)
        params = lambd
    elif distribution_type == 'Normal':
        st.latex(r'f(x;\mu,\sigma) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}')

        mu = st.slider("Normal Parameter (mu)", -10.0, 10.0, 0.0, step=0.1)
        sigma = st.slider("Normal Parameter (sigma)", 0.01, 10.0, 1.0, step=0.01)
        params = (mu, sigma)
    elif distribution_type == 'Gamma':
        st.latex(r'f(x;k,\theta) = \frac{1}{\Gamma(k)\theta^k} x^{k-1} e^{-\frac{x}{\theta}}, \quad x \geq 0')

        k = st.slider("Gamma Parameter (k)", 0.1, 10.0, 2.0, step=0.1)
        theta = st.slider("Gamma Parameter (theta)", 0.01, 10.0, 1.0, step=0.01)
        params = (k, theta)

    if params:
        analyzer = DistributionAnalyzer(distribution_type, params)

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
                y_pdf = weibull_min.pdf(x, *params)
                ax.plot(x, y_pdf, label='PDF', color='r')
            elif distribution_type == 'Geometric':
                y_pdf = geom.pmf(x, params)
                ax.plot(x, y_pdf, label='PMF', color='r')
            elif distribution_type == 'Bernoulli':
                y_pdf = bernoulli.pmf(x, params)
                ax.plot(x, y_pdf, label='PMF', color='r')
            elif distribution_type == 'Binomial':
                y_pdf = binom.pmf(x, params[0],params[1])
                ax.plot(x, y_pdf, label='PMF', color='r')
            elif distribution_type == 'Exponential':
                y_pdf = expon.pdf(x, scale=params)
                ax.plot(x, y_pdf, label='PDF', color='r')
            elif distribution_type == 'Normal':
                y_pdf = norm.pdf(x, loc=params[0], scale=params[1])
                ax.plot(x, y_pdf, label='PDF', color='r')
            elif distribution_type == 'Gamma':
                y_pdf = gamma.pdf(x, k, scale=params[1])
                ax.plot(x, y_pdf, label='PDF', color='r')
        if add_cdf:
            if distribution_type == 'Weibull':
                y_cdf = weibull_min.cdf(x, *params)
                ax.plot(x, y_cdf, label='CDF', color='b')
            elif distribution_type == 'Geometric':
                y_cdf = geom.cdf(x, params)
                ax.plot(x, y_cdf, label='CDF', color='b')
            elif distribution_type == 'Bernoulli':
                y_cdf = bernoulli.cdf(x, params)
                ax.plot(x, y_cdf, label='CDF', color='b')
            elif distribution_type == 'Binomial':
                y_cdf = binom.cdf(x, params[0],params[1])
                ax.plot(x, y_cdf, label='CDF', color='b')
            elif distribution_type == 'Exponential':
                y_cdf = expon.cdf(x, scale=params)
                ax.plot(x, y_cdf, label='CDF', color='b')
            elif distribution_type == 'Normal':
                y_cdf = norm.cdf(x, loc=params[0], scale=params[1])
                ax.plot(x, y_cdf, label='CDF', color='b')
            elif distribution_type == 'Gamma':
                y_cdf = gamma.cdf(x, k, scale=params[1])
                ax.plot(x, y_cdf, label='CDF', color='b')
        # Add legend
        ax.legend()

        # Display the plot using Streamlit
        st.pyplot(fig)
