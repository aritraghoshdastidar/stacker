import threading
import logging
import time
import uuid

from colorama.ansi import Fore

from .exceptions import (
    GraphError,
)
from .dag import DAG, DAGValidationError
from .status import (
    PENDING,
    SUBMITTED,
    COMPLETE,
    SKIPPED,
    CANCELLED
)

logger = logging.getLogger(__name__)


class Step(object):
    """State machine for executing generic actions related to stacks.

    Args:
        stack (:class:`stacker.stack.Stack`): the stack associated
            with this step

    """

    def __init__(self, stack):
        self.stack = stack
        self.status = PENDING
        self.last_updated = time.time()

    def __repr__(self):
        return "<stacker.plan.Step:%s>" % (self.stack.fqn,)

    @property
    def name(self):
        return self.stack.fqn

    @property
    def requires(self):
        return self.stack.requires

    @property
    def completed(self):
        """Returns True if the step is in a COMPLETE state."""
        return self.status == COMPLETE

    @property
    def skipped(self):
        """Returns True if the step is in a SKIPPED state."""
        return self.status == SKIPPED

    @property
    def cancelled(self):
        """Returns True if the step is in a CANCELLED state."""
        return self.status == CANCELLED

    @property
    def done(self):
        """Returns True if the step is finished (either COMPLETE, SKIPPED or
        CANCELLED)
        """
        return self.completed or self.skipped or self.cancelled

    @property
    def ok(self):
        """Returns True if the step is finished (either COMPLETE or SKIPPED)"""
        return self.completed or self.skipped

    @property
    def submitted(self):
        """Returns True if the step is SUBMITTED, COMPLETE, or SKIPPED."""
        return self.status >= SUBMITTED

    def set_status(self, status):
        """Sets the current step's status.

        Args:
            status (:class:`Status <Status>` object): The status to set the
                step to.
        """
        if status is not self.status:
            logger.debug("Setting %s state to %s.", self.stack.name,
                         status.name)
            self.status = status
            self.last_updated = time.time()

    def complete(self):
        """A shortcut for set_status(COMPLETE)"""
        self.set_status(COMPLETE)

    def skip(self):
        """A shortcut for set_status(SKIPPED)"""
        self.set_status(SKIPPED)

    def submit(self):
        """A shortcut for set_status(SUBMITTED)"""
        self.set_status(SUBMITTED)


class Plan(object):
    """A collection of :class:`Step` objects to execute.

    The :class:`Plan` helps organize the steps needed to execute a particular
    action for a set of :class:`stacker.stack.Stack` objects. When initialized
    with a set of steps, it will first build a Directed Acyclic Graph from the
    steps and their dependencies.

    Args:
        description (str): description of the plan
        steps (list): a list of :class:`Step` objects to execute.
        reverse (bool, optional): by default, the plan will be run in
            topological order based on each steps dependencies. Put
            more simply, the steps with no dependencies will be ran
            first. When this flag is set, the plan will be executed
            in reverse order.

    """

    def __init__(self, description, steps=None, reverse=False):
        self.description = description
        self.steps = {step.name: step for step in steps}
        self.dag = build_dag(steps)
        if reverse:
            self.dag = self.dag.transpose()
        self.id = uuid.uuid4()

    def execute(self, fn, **kwargs):
        """Executes the plan by walking the graph.

        Args:
            fn (func): a function that will be executed for each step. The
                function will be called multiple times until the step is
                `done`. The function should return a :class:`Status` each time
                it is called.

        """

        lock = threading.Lock()

        def check_point():
            lock.acquire()
            self._check_point()
            lock.release()

        check_point()

        def step_func(step):
            while not step.done:
                last_status = step.status
                status = fn(step)
                step.set_status(status)
                if status != last_status:
                    check_point()
            return step.ok

        return self.walk(step_func, **kwargs)

    def walk(self, step_func, semaphore=None):
        """Walks each step in the underlying graph, in topological order.

        Args:
            step_func (func): a function that will be called with the step.
            semaphore (threading.Semaphore, option): a semaphore object which
                can be used to control how many steps are executed in parallel.
                By default, there is not limit to the amount of parallelism,
                other than what the graph topology allows.

        """

        if not semaphore:
            semaphore = UnlimitedSemaphore()

        def walk_func(step_name):
            step = self.steps[step_name]
            semaphore.acquire()
            try:
                return step_func(step)
            finally:
                semaphore.release()

        return self.dag.walk(walk_func)

    def keys(self):
        return [k for k in self.steps]

    def outline(self, level=logging.INFO, message=""):
        pass

    def _check_point(self):
        """Outputs the current status of all steps in the plan."""
        status_to_color = {
            SUBMITTED.code: Fore.YELLOW,
            COMPLETE.code: Fore.GREEN,
        }
        logger.info("Plan Status:", extra={"reset": True, "loop": self.id})

        longest = 0
        messages = []

        nodes = self.dag.topological_sort()
        nodes.reverse()
        for step_name in nodes:
            step = self.steps[step_name]

            length = len(step.name)
            if length > longest:
                longest = length

            msg = "%s: %s" % (step.name, step.status.name)
            if step.status.reason:
                msg += " (%s)" % (step.status.reason)

            messages.append((msg, step))

        for msg, step in messages:
            parts = msg.split(' ', 1)
            fmt = "\t{0: <%d}{1}" % (longest + 2,)
            color = status_to_color.get(step.status.code, Fore.WHITE)
            logger.info(fmt.format(*parts), extra={
                'loop': self.id,
                'color': color,
                'last_updated': step.last_updated,
            })


def build_dag(steps):
    """Builds a Directed Acyclic Graph, given a list of steps.

    Args:
        steps (list): a list of :class:`Step` objects to execute.

    """

    dag = DAG()

    for step in steps:
        dag.add_node(step.name)

    for step in steps:
        for dep in step.requires:
            try:
                dag.add_edge(step.name, dep)
            except KeyError as e:
                raise GraphError(e, step.name, dep)
            except DAGValidationError as e:
                raise GraphError(e, step.name, dep)

    return dag


class UnlimitedSemaphore(object):
    """UnlimitedSemaphore implements the same interface as threading.Semaphore,
    but acquire's always succeed.
    """

    def acquire(self, *args):
        pass

    def release(self):
        pass
