require "rbsearch/version"
require "pycall"

PyCall.sys.path.append(__dir__)

puts "included"

module Rbsearch

  def self.foo
    puts "test"
    math = PyCall.import_module("math")
    puts math.sin(1.9 / 4) - Math.sin(Math::PI / 4)
    # Perform python nearest neighbors search in numpy
    nn = PyCall.import_module("vector.exact_nn")
    nn.main(["cat", "dog"])

    # foo.fun_stuff(5.0)
    # require 'pry'; binding.pry
  end

  class Error < StandardError; end
  # Your code goes here...
end
