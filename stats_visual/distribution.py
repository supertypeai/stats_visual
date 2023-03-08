import scipy.stats as stats
import numpy as np
import plotly.express as px
import plotly.io as pio

# Create a custom theme and set it as default
pio.templates["custom"] = pio.templates["plotly_white"]
pio.templates["custom"].layout.margin = {"b": 25, "l": 25, "r": 25, "t": 50}
pio.templates["custom"].layout.width = 800
pio.templates["custom"].layout.height = 600
pio.templates["custom"].layout.autosize = False
pio.templates["custom"].layout.font.update(
    {"family": "Arial", "size": 12, "color": "#707070"}
)
pio.templates["custom"].layout.title.update(
    {
        "xref": "container",
        "yref": "container",
        "font_size": 16,
        "y": 0.95,
        "font_color": "#353535",
        "x": 0.5,
    }
)
pio.templates["custom"].layout.xaxis.update(
    {"showline": True, "linecolor": "lightgray", "title_font_size": 14}
)
pio.templates["custom"].layout.yaxis.update(
    {"showline": True, "linecolor": "lightgray", "title_font_size": 14}
)
pio.templates["custom"].layout.colorway = [
    "#1F77B4",
    "#FF7F0E",
    "#54A24B",
    "#D62728",
    "#C355FA",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#FFE323",
    "#17BECF",
]
pio.templates.default = "custom"


class _DiscreteDist:
    def __init__(self):
        """Base class of discrete distributions for plotting the probability distribution
        """
        # Set the x and y values for the probability distribution
        x_upper_limit = self._calc_x_limit()
        self._x_vals = np.arange(0, x_upper_limit + 1, 1)
        self._y_vals = np.array([self.calc_pmf(x, plot=False) for x in self._x_vals])

    def plot_dist(self, title: str):
        """Plot the probability distribution for a discrete distribution
        Args:
            labels (dict): dictionary of labels for the x and y axes
            title (str): title of the plot
        Returns:
            plotly.graph_objects.Figure: plot of the distribution
        """
        fig = px.bar(
            x=self._x_vals,
            y=self._y_vals,
            title=title,
            labels={"x": "K", "y": "Probability"},
        )
        fig.update_traces(hovertemplate="P(K = %{x}) = %{y:.5f}<extra></extra>")
        return fig

    def calc_pmf(self):
        """Placeholder for calc_pmf"""
        pass

    def _highlight_pmf(self, fig, k: int, pmf: float, add_title: str):
        """Highlight the probability mass function at k
        Args:
            fig (plotly.graph_objects.Figure): plot of the distribution
            k (int): number of successes
            pmf (float): probability mass function at k
            add_title (str): text to be appended to the existing plot title
        Returns:
            plotly.graph_objects.Figure: plot of the distribution with the probability mass function at k highlighted
        """
        # Highlight the probability mass function at k by assigning orange color to the k-th bar
        k_index = np.where(self._x_vals == k)[0]
        colors = np.array(["#1F77B4"] * self._x_vals.shape[0])
        colors[k_index] = ["#FF7F0E"]
        fig.data[0].marker.color = colors
        # Edit the title of the figure
        fig.layout.title.text += add_title
        return fig

    def calc_cum_p(self):
        """Placeholder for calc_cum_p"""
        pass

    def _highlight_cum_p(self, fig, k: int, cum_p: float, add_title: str):
        """Highlight the cumulative probability at k
        Args:
            fig (plotly.graph_objects.Figure): plot of the distribution
            k (int): number of successes
            cum_p (float): cumulative probability, P(K <= k)
            add_title (str): text to be appended to the existing plot title
        Returns:
            plotly.graph_objects.Figure: plot of the distribution with cumulative probability at k highlighted
        """
        # Highlight the cumulative probability at k by assigning orange color to the first k-th bars
        k_index = np.where(self._x_vals <= k)[0]
        colors = np.array(["#1F77B4"] * self._x_vals.shape[0])
        colors[k_index] = ["#FF7F0E"]
        fig.data[0].marker.color = colors
        # Edit the title of the figure
        fig.layout.title.text += add_title
        return fig

    def _calc_x_limit(self):
        """Placeholder for _calc_x_limit"""
        pass

    def __repr__(self):
        """Returns the representation of the class
        Returns:
            str: representation of the class
        """
        return "Base class for discrete distribution"


class BinomialDist(_DiscreteDist):
    def __init__(self, n: int, p: float):
        """Binomial distribution class for plotting probability distribution and calculating probability mass function/ cumulative probability
        Args:
            n (int): number of trials
            p (float): probability of success
        Raises:
            AssertionError: probability of success (p) must be between 0 and 1
            AssertionError: number of trials (n) must be greater than 0
        """
        if p < 0 or p > 1:
            raise AssertionError("probability of success (p) must be between 0 and 1")
        elif n <= 0:
            raise AssertionError("number of trials (n) must be greater than 0")
        self._n = n
        self._p = p
        # Set the x and y values for the probability distribution
        super().__init__()

    @property
    def n(self):
        """Get the number of trials for the binomial distribution
        Returns:
            int: number of trials for the binomial distribution
        """
        return self._n

    @n.setter
    def n(self, new_n: int):
        """Set a new number of trials for the binomial distribution
        Args:
            new_n (int): new number of trials for the binomial distribution
        Raises:
            AssertionError: the new number of trials (new_n) must be greater than 0
        """
        if new_n <= 0:
            raise AssertionError(
                "the new number of trials (new_n) must be greater than 0"
            )
        self._n = new_n
        # Set the x and y values for the probability distribution
        super().__init__()

    @property
    def p(self):
        """Get the probability of success for the binomial distribution
        Returns:
            float: probability of success for the binomial distribution
        """
        return self._p

    @p.setter
    def p(self, new_p: float):
        """Set a new probability of success for the binomial distribution
        Args:
            new_p (float): new probability of success for the binomial distribution
        Raises:
            AssertionError: the new probability of success (new_p) must be between 0 and 1
        """
        if new_p < 0 or new_p > 1:
            raise AssertionError(
                "the new probability of success (new_p) must be between 0 and 1"
            )
        self._p = new_p
        # Set the x and y values for the probability distribution
        super().__init__()

    def plot_dist(self):
        """Plot the probability distribution for a binomial distribution
        Returns:
            plotly.graph_objects.Figure: plot of the distribution
        """
        title = "Probability Mass Function for Binomial Distribution<br>n = {}, p = {}".format(
            self._n, self._p
        )
        fig = super().plot_dist(title)
        return fig

    def calc_pmf(self, k: int, plot=True):
        """Calculate the probability mass function for a binomial distribution and optionally plot the distribution
        Args:
            k (int): number of successes
            plot (bool, optional): if True, return a plot of the distribution with probability mass funtion at k highlighted. Defaults to True.
        Raises:
            AssertionError: the number of successes (k) should be between 0 and n
        Returns:
            plotly.graph_objects.Figure or float: plot of the distribution with probability mass function at k highlighted or probability mass function at k 
        """
        if k < 0 or k > self._n:
            raise AssertionError(
                "the number of successes (k) should be between 0 and {} (n)".format(
                    self._n
                )
            )
        pmf = stats.binom.pmf(k=k, n=self._n, p=self._p)
        if plot == False:
            return pmf
        elif plot == True:
            fig = self.plot_dist()
            add_title = ", <span style='color:#FF7F0E'><b>P(K = {}) = {:.5f}</b></span>".format(
                k, pmf
            )
            fig = super()._highlight_pmf(fig, k, pmf, add_title)
            return fig

    def calc_cum_p(self, k: int, plot=True):
        """Calculate the cumulative probability for a binomial distribution and optionally plot the distribution
        Args:
            k (int): number of successes
            plot (bool, optional): if True, return a plot of the distribution with cumulative probability at k highlighted. Defaults to True.
        Raises:
            AssertionError: the number of successes (k) should be between 0 and n
        Returns:
            plotly.graph_objects.Figure or float: plot of the distribution with cumulative probability at k highlighted or cumulative probability at k
        """
        if k < 0 or k > self._n:
            raise AssertionError(
                "the number of successes (k) should be between 0 and {} (n)".format(
                    self._n
                )
            )
        cum_p = stats.binom.cdf(k=k, n=self._n, p=self._p)
        if plot == False:
            return cum_p
        elif plot == True:
            fig = self.plot_dist()
            add_title = ", <span style='color:#FF7F0E'><b>P(K <= {}) = {:.5f}</b></span>".format(
                k, cum_p
            )
            fig = super()._highlight_cum_p(fig, k, cum_p, add_title)
            return fig

    def _calc_x_limit(self):
        """Calculate the x-axis upper limit for the distribution
        Returns:
            int: x-axis upper limit
        """
        return self._n

    def __repr__(self):
        """Returns the representation of the class
        Returns:
            str: representation of the class
        """
        return "Binomial distribution with number of trials (n) = {} and probability of success (p) = {}".format(
            self._n, self._p
        )


class PoissonDist(_DiscreteDist):
    def __init__(self, mu: int):
        """Poisson distribution class for plotting probability distribution and calculating probability mass function/ cumulative probability
        Args:
            mu (int): mean number of occurences over a given interval
        Raises:
            AssertionError: the mean number of occurences over a given interval (mu) must be greater than 0
        """
        if mu <= 0:
            raise AssertionError(
                "the mean number of occurences over a given interval (mu) must be greater than 0"
            )
        self._mu = mu
        # Set the x and y values for the probability distribution
        super().__init__()

    @property
    def mu(self):
        """Get the mean number of occurences over a given interval for the poisson distribution
        Returns:
            int: mean number of occurences over a given interval for the poisson distribution
        """
        return self._mu

    @mu.setter
    def mu(self, new_mu: int):
        """Set a new mean number of occurences over a given interval for the poisson distribution
        Args:
            new_mu (int): mean number of occurences over a given interval for the poisson distribution
        Raises:
            AssertionError: the new mean number of occurences over a given interval (new_mu) must be greater than 0
        """
        if new_mu <= 0:
            raise AssertionError(
                "the new mean number of occurences over a given interval (new_mu)  must be greater than 0"
            )
        self._mu = new_mu
        # Set the x and y values for the probability distribution
        super().__init__()

    def plot_dist(self):
        """Plot the probability distribution for a poisson distribution
        Returns:
            plotly.graph_objects.Figure: plot of the distribution
        """
        title = "Probability Mass Function for Poisson Distribution<br>µ = {}".format(
            self._mu
        )
        fig = super().plot_dist(title)
        return fig

    def calc_pmf(self, k: int, plot=True):
        """Calculate the probability mass function for a poisson distribution and optionally plot the distribution
        Args:
            k (int): number of occurences
            plot (bool, optional): if True, return a plot of the distribution with probability mass funtion at k highlighted. Defaults to True.
        Raises:
            AssertionError: the number of occurences (k) should be greater than or equal to 0
        Returns:
            plotly.graph_objects.Figure or float: plot of the distribution with probability mass function at k highlighted or probability mass function at k
        """
        if k < 0:
            raise AssertionError(
                "the number of occurences (k) should be greater than or equal to 0"
            )
        pmf = stats.poisson.pmf(k=k, mu=self._mu)
        if plot == False:
            return pmf
        elif plot == True:
            fig = self.plot_dist()
            add_title = ", <span style='color:#FF7F0E'><b>P(K = {}) = {:.5f}</b></span>".format(
                k, pmf
            )
            fig = super()._highlight_pmf(fig, k, pmf, add_title)
            return fig

    def calc_cum_p(self, k: int, plot=True):
        """Calculate the cumulative probability for a poisson distribution and optionally plot the distribution
        Args:
            k (int): number of occurences
            plot (bool, optional): if True, return a plot of the distribution with cumulative probability at k highlighted. Defaults to True.
        Raises:
            AssertionError: the number of occurences (k) should be greater than or equal to 0
        Returns:
            plotly.graph_objects.Figure or float: plot of the distribution with cumulative probability at k highlighted or cumulative probability at k
        """
        if k < 0:
            raise AssertionError(
                "the number of occurences (k) should be greater than or equal to 0"
            )
        cum_p = stats.poisson.cdf(k=k, mu=self._mu)
        if plot == False:
            return cum_p
        elif plot == True:
            fig = self.plot_dist()
            add_title = ", <span style='color:#FF7F0E'><b>P(K <= {}) = {:.5f}</b></span>".format(
                k, cum_p
            )
            fig = super()._highlight_cum_p(fig, k, cum_p, add_title)
            return fig

    def _calc_x_limit(self):
        """Calculate the x-axis upper limit for the distribution
        Returns:
            int: x-axis upper limit
        """
        k = self._mu
        # Only include k with a probability mass function greater than 0.001 in the x-axis
        while self.calc_pmf(k, plot=False) > 0.001:
            k += 1
        return k - 1

    def __repr__(self):
        """Returns the representation of the class
        Returns:
            str: representation of the class
        """
        return "Poisson distribution with mean number of occurences over a given interval (µ) = {}".format(
            self._mu
        )


class _ContinuousDist:
    def __init__(self):
        """Base class of continuous distribution for plotting the probability distribution
        """
        # Set the x and y values for the probability distribution
        x_lower_limit, x_upper_limit = self._calc_x_limit()
        self._x_vals = np.linspace(x_lower_limit, x_upper_limit, 10000)
        self._y_vals = np.array([self.calc_pdf(x, plot=False) for x in self._x_vals])

    def plot_dist(self, title: str):
        """Plot the probability distribution for a continuous distribution
        Args:
            title (str): title of the plot
        Returns:
            plotly.graph_objects.Figure: plot of the distribution
        """
        fig = px.line(
            x=self._x_vals,
            y=self._y_vals,
            labels={"x": "X", "y": "Probability Density"},
            title=title,
        )
        fig.update_yaxes(rangemode="tozero")
        fig.update_traces(hovertemplate="F(X = %{x:.3f}) = %{y:.5f}<extra></extra>")
        return fig

    def calc_pdf(self):
        """Placeholder for calc_pdf"""
        pass

    def _highlight_pdf(self, fig, x: float, pdf: float, add_title: str):
        """Highlight the probability density function at x
        Args:
            fig (plotly.graph_objects.Figure): plot of the distribution
            x (float): value of x
            pdf (float): probability density function at x
            add_title (str): text to be appended to the existing plot title
        Returns:
            plotly.graph_objects.Figure: plot of the distribution with the probability density function at x highlighted
        """
        # Highlight the probability density function at x by adding a orange marker
        fig.add_scatter(
            x=[x],
            y=[pdf],
            mode="markers",
            marker_size=10,
            marker_color="rgba(255, 127, 14, 1)",
            showlegend=False,
        )
        fig.update_traces(hovertemplate="F(X = %{x}) = %{y:.5f}<extra></extra>")
        # Edit the figure title
        fig.layout.title.text += add_title
        return fig

    def calc_cum_p(self):
        """Placeholder for calc_cum_p"""
        pass

    def _highlight_cum_p(self, fig, x: float, cum_p: "float", add_title: str):
        """Highlight the cumulative probability at x
        Args:
            fig (plotly.graph_objects.Figure): plot of the distribution
            x (float): value of x
            cum_p (float): cumulative probability at x
            add_title (str): text to be appended to the existing plot title
        Returns:
            plotly.graph_objects.Figure or float: plot of the distribution with cumulative probability at x highlighted or cumulative probability at x
        """
        # Highlight the cumulative probability at x by filling the area under the curve to the left of x with orange color
        fig.add_scatter(
            x=self._x_vals[self._x_vals < x],
            y=self._y_vals,
            fill="tozeroy",
            mode="none",
            fillcolor="rgba(255, 127, 14, 0.4)",
            showlegend=False,
        )
        fig.update_traces(hovertemplate="F(X = %{x}) = %{y:.5f}<extra></extra>")
        # Edit the figure title
        fig.layout.title.text += add_title
        return fig

    def _calc_x_limit(self):
        """Placeholder for _calc_x_limit"""
        pass

    def __repr__(self):
        """Returns the string representation of the class
        Returns:
            str: representation of the class
        """
        return "Base class for continuous distribution"


class NormalDist(_ContinuousDist):
    def __init__(self, mu: float, sigma: float):
        """Normal distribution class for plotting probability distribution and calculating probability density function/ cumulative probability
        Args:
            mu (float): mean of the distribution
            sigma (float): standard deviation of the distribution
        
        Raises:
            AssertionError: the standard deviation must be greater than 0
        """
        if sigma < 0:
            raise AssertionError("the standard deviation must be greater than 0")
        self._mu = mu
        self._sigma = sigma
        # Set the x and y values for the probability distribution
        super().__init__()

    @property
    def mu(self):
        """Get the mean of the normal distribution
        Returns:
            float: mean of the normal distribution
        """
        return self._mu

    @mu.setter
    def mu(self, new_mu: float):
        """Set a new mean for the normal distribution
        Args:
            new_mu (float): new mean of the normal distribution
        """
        self._mu = new_mu
        # Set the x and y values for the probability distribution
        super().__init__()

    @property
    def sigma(self):
        """Get the standard deviation of the normal distribution
        Returns:
            float: standard deviation of the normal distribution
        """
        return self._sigma

    @sigma.setter
    def sigma(self, new_sigma):
        """Set a new standard deviation for the normal distribution
        Args:
            new_sigma (float): new standard deviation of the normal distribution
        
        Raises:
            AssertionError: the new standard deviation must be greater than 0
        """
        if new_sigma < 0:
            raise AssertionError("the new standard deviation must be greater than 0")
        self._sigma = new_sigma
        # Set the x and y values for the probability distribution
        super().__init__()

    def plot_dist(self):
        """Plot the probability distribution for a normal distribution
        Returns:
            plotly.graph_objects.Figure: plot of the distribution
        """
        title = "Probability Density Function for Normal Distribution<br>µ = {}, σ = {}".format(
            self._mu, self._sigma
        )
        fig = super().plot_dist(title)
        return fig

    def calc_pdf(self, x: float, plot=True):
        """Calculate the probability density function for a normal distribution and optionally plot the distribution
        Args:
            x (float): value of x
            plot (bool, optional): if True, return a plot of the distribution with probability density function at x highlighted. Defaults to True.
        Returns:
            plotly.graph_objects.Figure or float: plot of the distribution with probability density function at x highlighted or probability density function at x
        """
        pdf = stats.norm.pdf(x=x, loc=self._mu, scale=self._sigma)
        if plot == False:
            return pdf
        elif plot == True:
            fig = self.plot_dist()
            add_title = ", <span style='color:#FF7F0E'><b>F(X = {}) = {:.5f}</b></span>".format(
                x, pdf
            )
            fig = super()._highlight_pdf(fig, x, pdf, add_title)
            return fig

    def calc_cum_p(self, x: float, plot=True):
        """Calculate the cumulative probability for a normal distribution and optionally plot the distribution
        Args:
            x (float): value of x
            plot (bool, optional): if True, return a plot of the distribution with cumulative probability at x highlighted. Defaults to True.
        Returns:
            plotly.graph_objects.Figure or float: plot of the distribution with cumulative probability at x highlighted or cumulative probability at x
        """
        cum_p = stats.norm.cdf(x=x, loc=self._mu, scale=self._sigma)
        if plot == False:
            return cum_p
        elif plot == True:
            fig = self.plot_dist()
            add_title = ", <span style='color:#FF7F0E'><b>P(X < {}) = {:.5f}</b></span>".format(
                x, cum_p
            )
            fig = super()._highlight_cum_p(fig, x, cum_p, add_title)
            return fig

    def _calc_x_limit(self):
        """Calculate the x-axis lower and upper limit for the distribution
        Returns:
            tuple: x-axis lower and upper limit
        """
        # Set the x-axis limit to 4.5 standard deviations away from the mean (as they will cover most of the distribution)
        x_lower_limit = self._mu - (4.5 * self._sigma)
        x_upper_limit = self._mu + (4.5 * self._sigma)
        return (x_lower_limit, x_upper_limit)

    def __repr__(self):
        """Returns the string representation of the class
        Returns:
            str: representation of the class
        """
        return "Normal distribution with mean (µ) = {} and standard deviation (σ) = {}".format(
            self._mu, self._sigma
        )


class StudentsTDist(_ContinuousDist):
    def __init__(self, df: int):
        """Student's t distribution class for plotting probability distribution and calculating probability density function/ cumulative probability
        Args:
            df (int): degree of freedom of the distribution
        Raises:
            AssertionError: the degree of freedom (df) must be greater than 0
        """
        if df < 1:
            raise AssertionError("the degree of freedom (df) must be greater than 0")
        self._df = df
        # Set the x and y values for the probability distribution
        super().__init__()

    @property
    def df(self):
        """Get the degree of freedom of the Student's t distribution
        Returns:
            int: degree of freedom of the Student's t distribution
        """
        return self._df

    @df.setter
    def df(self, new_df):
        """Set a new degree of freedom for the Student's t distribution
        Args:
            new_df (int): new degree of freedom of the Student's t distribution
        Raises:
            AssertionError: the new degree of freedom (new_df) must be greater than 0
        """
        if new_df < 1:
            raise AssertionError(
                "the new degree of freedom (new_df) must be greater than 0"
            )
        self._df = new_df
        # Set the x and y values for the probability distribution
        super().__init__()

    def plot_dist(self):
        """Plot the probability distribution for a Student's t distribution
        Returns:
            plotly.graph_objects.Figure: plot of the distribution
        """
        title = "Probability Density Function for Student's T Distribution<br>df = {}".format(
            self._df
        )
        fig = super().plot_dist(title)
        return fig

    def calc_pdf(self, x: float, plot=True):
        """Calculate the probability density function for a Student's t distribution and optionally plot the distribution
        Args:
            x (float): value of x
            plot (bool, optional): if True, return a plot of the distribution with probability density function at x highlighted. Defaults to True.
        Returns:
            plotly.graph_objects.Figure or float: plot of the distribution with probability density function at x highlighted or probability density function at x
        """
        pdf = stats.t.pdf(x=x, df=self._df)
        if plot == False:
            return pdf
        elif plot == True:
            fig = self.plot_dist()
            add_title = ", <span style='color:#FF7F0E'><b>F(X = {}) = {:.5f}</b></span>".format(
                x, pdf
            )
            fig = super()._highlight_pdf(fig, x, pdf, add_title)
            return fig

    def calc_cum_p(self, x: float, plot=True):
        """Calculate the cumulative probability for a Student's t distribution and optionally plot the distribution
        Args:
            x (float): value of x
            plot (bool, optional): if True, return a plot of the distribution with cumulative probability at x highlighted. Defaults to True.
        Returns:
            plotly.graph_objects.Figure or float: plot of the distribution with cumulative probability at x highlighted or cumulative probability at x
        """
        cum_p = stats.t.cdf(x=x, df=self._df)
        if plot == False:
            return cum_p
        elif plot == True:
            fig = self.plot_dist()
            add_title = ", <span style='color:#FF7F0E'><b>P(X < {}) = {:.5f}</b></span>".format(
                x, cum_p
            )
            fig = super()._highlight_cum_p(fig, x, cum_p, add_title)
            return fig

    def _calc_x_limit(self):
        """Calculate the x-axis lower and upper limit for the distribution
        Returns:
            tuple: x-axis lower and upper limit 
        """
        x = 0
        # Only include x with a probability density function greater than 0.001 in the x-axis
        while self.calc_pdf(x, plot=False) > 0.001:
            x += 0.5
        return (-x, x)

    def __repr__(self):
        """Returns the representation of the class
        Returns:
            str: representation of the class
        """
        return "Student's T distribution with degree of freedom (df) = {}".format(
            self._df
        )
